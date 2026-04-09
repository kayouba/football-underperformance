"""
Module de modelisation avancee.
Regression logistique, analyse de survie, null model Bernoulli.
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats
from tqdm import tqdm

from src.utils import load_config, setup_logger

logger = setup_logger(__name__)


# ===============================================================
# REGRESSION LOGISTIQUE TEMPORELLE
# ===============================================================

class TemporalLogisticModel:
    """
    Regression logistique :
    P(conceder dans 10 min) ~ underperf + score + controls

    Usage :
        model = TemporalLogisticModel()
        result = model.fit(timeline_df, window=10)
        model.print_results(result)
        cv = model.cross_validate(timeline_df)
    """

    def __init__(self, config=None):
        self.config = config or load_config()

    def fit(self, timeline_df, window=10):
        """
        Fit une regression logistique avec clustered SE par match.

        Retourne le resultat statsmodels.
        """
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import Logit

        col_y = f'future_conceded_{window}min'
        df = timeline_df.copy()

        # Terme quadratique
        df['minute_sq'] = df['minute'] ** 2

        # Dummies score state
        dummies = pd.get_dummies(df['score_state'], prefix='state', drop_first=True)
        df = pd.concat([df, dummies], axis=1)

        # Features
        feature_cols = ['cum_underperf', 'score_diff', 'minute', 'minute_sq']

        # Ajouter dummies (garder celles qui existent)
        for col in dummies.columns:
            if col in df.columns:
                feature_cols.append(col)

        # Rolling features
        for col in ['rolling_xg_10min', 'rolling_underperf_10min',
                     'shots_since_last_goal', 'xg_since_last_goal']:
            if col in df.columns:
                feature_cols.append(col)

        # Strength si disponible
        for col in ['strength_diff', 'team_xg_per90', 'opp_xg_per90']:
            if col in df.columns:
                feature_cols.append(col)

        # Nettoyer
        cols_needed = feature_cols + [col_y, 'match_id']
        df_clean = df[cols_needed].dropna()

        X = df_clean[feature_cols].astype(float)
        y = df_clean[col_y].astype(float)
        groups = df_clean['match_id'].reset_index(drop=True)

        X_const = sm.add_constant(X)

        # Fit avec clustered SE par match
        try:
            model = Logit(y, X_const)
            result = model.fit(disp=0, cov_type='cluster',
                               cov_kwds={'groups': groups})
        except Exception as e:
            logger.warning(f"Clustered SE echoue ({e}), fallback standard")
            model = Logit(y, X_const)
            result = model.fit(disp=0)

        logger.info(f"Logit fit OK : {len(df_clean)} obs, "
                     f"pseudo-R2={result.prsquared:.4f}")

        return result

    def fit_incremental(self, timeline_df, window=10):
        """
        Fit des modeles incrementaux pour voir l'apport de chaque variable.
        M1 : underperf seul
        M2 : + score_diff
        M3 : + minute
        M4 : + rolling features
        """
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import Logit

        col_y = f'future_conceded_{window}min'
        df = timeline_df.copy()
        df['minute_sq'] = df['minute'] ** 2

        specs = {
            'M1: underperf seul': ['cum_underperf'],
            'M2: + score': ['cum_underperf', 'score_diff'],
            'M3: + temps': ['cum_underperf', 'score_diff', 'minute', 'minute_sq'],
            'M4: + rolling': ['cum_underperf', 'score_diff', 'minute', 'minute_sq',
                              'rolling_xg_10min', 'shots_since_last_goal'],
        }

        results = {}
        for name, features in specs.items():
            features = [f for f in features if f in df.columns]
            df_clean = df[features + [col_y]].dropna()
            X = sm.add_constant(df_clean[features])
            y = df_clean[col_y]

            try:
                model = Logit(y, X)
                res = model.fit(disp=0)
                results[name] = {
                    'result': res,
                    'aic': res.aic,
                    'bic': res.bic,
                    'pseudo_r2': res.prsquared,
                    'n_obs': len(df_clean),
                    'coef_underperf': res.params.get('cum_underperf', np.nan),
                    'pval_underperf': res.pvalues.get('cum_underperf', np.nan),
                }
            except Exception as e:
                logger.warning(f"{name}: erreur {e}")
                results[name] = {'error': str(e)}

        return results

    def cross_validate(self, timeline_df, window=10, n_splits=5):
        """
        Validation croisee GroupKFold par match.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import roc_auc_score, brier_score_loss

        col_y = f'future_conceded_{window}min'

        feature_cols = ['cum_underperf', 'score_diff', 'minute',
                        'rolling_xg_10min', 'shots_since_last_goal']
        feature_cols = [f for f in feature_cols if f in timeline_df.columns]

        df = timeline_df.dropna(subset=feature_cols + [col_y])
        X = df[feature_cols].values
        y = df[col_y].values
        groups = df['match_id'].values

        gkf = GroupKFold(n_splits=n_splits)
        aucs, briers = [], []

        for train_idx, test_idx in gkf.split(X, y, groups):
            model = LogisticRegression(max_iter=1000, solver='lbfgs')
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict_proba(X[test_idx])[:, 1]

            if len(np.unique(y[test_idx])) > 1:
                aucs.append(roc_auc_score(y[test_idx], y_pred))
            briers.append(brier_score_loss(y[test_idx], y_pred))

        return {
            'mean_auc': np.mean(aucs) if aucs else np.nan,
            'std_auc': np.std(aucs) if aucs else np.nan,
            'mean_brier': np.mean(briers),
            'std_brier': np.std(briers),
            'n_folds': n_splits
        }

    @staticmethod
    def print_results(result):
        """Affiche les resultats de la regression."""
        print("\n" + "=" * 65)
        print("  REGRESSION LOGISTIQUE")
        print("=" * 65)
        print(f"  N obs         : {int(result.nobs)}")
        print(f"  Pseudo R2     : {result.prsquared:.4f}")
        print(f"  AIC           : {result.aic:.1f}")
        print(f"  Log-lik       : {result.llf:.1f}")
        print()

        # Coefficients
        summary_df = pd.DataFrame({
            'coef': result.params,
            'std_err': result.bse,
            'z': result.tvalues,
            'p_value': result.pvalues,
            'odds_ratio': np.exp(result.params)
        })

        print("  Coefficients :")
        print("  " + "-" * 63)
        for idx, row in summary_df.iterrows():
            sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
            print(f"  {idx:<28s} {row['coef']:>8.4f}  (SE={row['std_err']:.4f})  "
                  f"p={row['p_value']:.4f} {sig:3s}  OR={row['odds_ratio']:.4f}")
        print()


# ===============================================================
# ANALYSE DE SURVIE
# ===============================================================

class SurvivalAnalysis:
    """
    Analyse de survie : temps jusqu'au premier but encaisse
    apres un episode d'underperformance.

    Usage :
        sa = SurvivalAnalysis()
        surv_df = sa.prepare_survival_data(timeline, shots)
        cph = sa.fit_cox_model(surv_df)
        km = sa.kaplan_meier_by_group(surv_df)
    """

    def prepare_survival_data(self, timeline_df, shots_df, threshold=0.5):
        """
        Prepare les donnees de survie.
        Pour chaque match x equipe :
        - trigger = minute ou underperf >= threshold
        - event = 1 si but encaisse apres trigger, 0 si censure
        - duration = minutes entre trigger et event/fin de match
        """
        records = []

        for (match_id, team), group in timeline_df.groupby(['match_id', 'team']):
            group = group.sort_values('minute')

            # Trouver le trigger
            trigger_rows = group[group['cum_underperf'] >= threshold]
            if len(trigger_rows) == 0:
                continue

            trigger_minute = trigger_rows.iloc[0]['minute']
            opponent = trigger_rows.iloc[0]['opponent']
            max_minute = group['minute'].max()

            # Trouver les buts adverses apres le trigger
            opp_shots = shots_df[
                (shots_df['match_id'] == match_id) &
                (shots_df['team'] == opponent) &
                (shots_df['is_goal'] == 1) &
                (shots_df['game_minute'] > trigger_minute)
            ]

            if len(opp_shots) > 0:
                first_concede_minute = opp_shots['game_minute'].min()
                duration = first_concede_minute - trigger_minute
                event = 1
            else:
                duration = max_minute - trigger_minute
                event = 0

            records.append({
                'match_id': match_id,
                'team': team,
                'opponent': opponent,
                'trigger_minute': trigger_minute,
                'duration': max(duration, 0.5),
                'event': event,
                'cum_underperf': trigger_rows.iloc[0]['cum_underperf'],
                'score_state': trigger_rows.iloc[0]['score_state'],
                'score_diff': trigger_rows.iloc[0]['score_diff'],
                'is_home': trigger_rows.iloc[0]['is_home'],
            })

        df = pd.DataFrame(records)
        logger.info(f"Survie : {len(df)} episodes, "
                     f"{df['event'].sum()} events ({df['event'].mean()*100:.1f}%)")
        return df

    def fit_cox_model(self, survival_df):
        """Cox Proportional Hazards."""
        from lifelines import CoxPHFitter

        df = survival_df.copy()

        # Encoder score state
        dummies = pd.get_dummies(df['score_state'], prefix='state', drop_first=True)
        df = pd.concat([df, dummies], axis=1)

        covariates = ['cum_underperf', 'score_diff', 'trigger_minute', 'is_home']
        covariates += [c for c in dummies.columns]

        cols = ['duration', 'event'] + covariates
        df_fit = df[cols].dropna()

        cph = CoxPHFitter()
        cph.fit(df_fit, duration_col='duration', event_col='event',
                show_progress=False)

        logger.info(f"Cox PH fit OK : {len(df_fit)} obs, "
                     f"concordance={cph.concordance_index_:.3f}")
        return cph

    def kaplan_meier_by_group(self, survival_df, group_col='underperf_group'):
        """Courbes de Kaplan-Meier stratifiees."""
        from lifelines import KaplanMeierFitter

        df = survival_df.copy()
        df['underperf_group'] = pd.qcut(
            df['cum_underperf'], q=3,
            labels=['low', 'medium', 'high'],
            duplicates='drop'
        )

        results = {}
        for group_name, group_data in df.groupby('underperf_group', observed=True):
            if len(group_data) < 10:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(group_data['duration'],
                    event_observed=group_data['event'],
                    label=str(group_name))
            results[str(group_name)] = kmf

        return results

    def log_rank_test(self, survival_df):
        """Log-rank test entre groupes d'underperformance."""
        from lifelines.statistics import logrank_test

        df = survival_df.copy()
        median_underperf = df['cum_underperf'].median()

        high = df[df['cum_underperf'] >= median_underperf]
        low = df[df['cum_underperf'] < median_underperf]

        if len(high) < 5 or len(low) < 5:
            return {'p_value': np.nan, 'test_statistic': np.nan}

        result = logrank_test(
            high['duration'], low['duration'],
            event_observed_A=high['event'],
            event_observed_B=low['event']
        )

        return {
            'test_statistic': result.test_statistic,
            'p_value': result.p_value,
            'n_high': len(high),
            'n_low': len(low)
        }


# ===============================================================
# NULL MODEL BERNOULLI
# ===============================================================

class NullModelSimulation:
    """
    Simule un monde ou chaque tir est un Bernoulli(xG) independant.
    Compare l'effet observe a l'effet mecanique attendu.

    C'est LE test decisif du projet.

    Usage :
        null = NullModelSimulation()
        result = null.run(shots_df, observed_effect, n_simulations=1000)
        null.print_results(result)
    """

    def __init__(self, config=None):
        self.config = config or load_config()

    def run(self, shots_df, observed_effect, threshold=0.5,
            window=10, n_simulations=500, seed=42):
        """
        Pour chaque simulation :
        1. Resimule is_goal ~ Bernoulli(xG) pour CHAQUE tir
        2. Recalcule cum_underperf avec les nouveaux buts
        3. Recalcule les outcomes (future_conceded change aussi)
        4. Mesure la difference traite - controle

        Parameters
        ----------
        shots_df : DataFrame original de tirs
        observed_effect : float, la difference observee dans les donnees reelles
        threshold : seuil d'underperformance
        window : fenetre future en minutes
        n_simulations : nombre de simulations
        seed : random seed

        Returns
        -------
        dict avec distribution nulle et comparaison
        """
        rng = np.random.default_rng(seed)
        null_effects = []

        match_ids = shots_df['match_id'].unique()

        logger.info(f"Null model : {n_simulations} simulations, "
                     f"{len(match_ids)} matchs, seuil={threshold}")

        for sim in tqdm(range(n_simulations), desc="Null model"):
            sim_effect = self._simulate_one(
                shots_df, match_ids, rng, threshold, window
            )
            if sim_effect is not None:
                null_effects.append(sim_effect)

        null_effects = np.array(null_effects)

        # Comparaison
        p_value = (null_effects <= observed_effect).mean()

        excess = observed_effect - null_effects.mean()

        return {
            'observed_effect': observed_effect,
            'null_effects': null_effects,
            'null_mean': null_effects.mean(),
            'null_std': null_effects.std(),
            'null_median': np.median(null_effects),
            'ci_95': (np.percentile(null_effects, 2.5),
                      np.percentile(null_effects, 97.5)),
            'p_value': p_value,
            'excess_effect': excess,
            'n_simulations': len(null_effects),
            'threshold': threshold,
            'window': window,
            'percentile_of_observed': (null_effects <= observed_effect).mean() * 100
        }

    def _simulate_one(self, shots_df, match_ids, rng, threshold, window):
        """Une simulation complete."""

        all_diffs = []

        for match_id in match_ids:
            match_shots = shots_df[shots_df['match_id'] == match_id].copy()
            teams = match_shots['team'].unique()
            if len(teams) != 2:
                continue

            # Resimule les buts
            match_shots = match_shots.copy()
            match_shots['sim_goal'] = rng.binomial(1, match_shots['xg'].values)

            home_team = match_shots['home_team'].iloc[0]
            max_minute = int(min(match_shots['game_minute'].max() + 1, 95))

            for team in teams:
                opponent = [t for t in teams if t != team][0]

                t_shots = match_shots[match_shots['team'] == team]
                o_shots = match_shots[match_shots['team'] == opponent]

                t_min = t_shots['game_minute'].values
                t_xg = t_shots['xg'].values
                t_goals_sim = t_shots['sim_goal'].values

                o_min = o_shots['game_minute'].values
                o_goals_sim = o_shots['sim_goal'].values

                # Pour chaque minute, calculer underperf et outcome
                for t in range(1, min(max_minute - window + 1, 86)):
                    past_mask = t_min <= t
                    cum_xg = t_xg[past_mask].sum()
                    cum_goals = t_goals_sim[past_mask].sum()
                    cum_underperf = cum_xg - cum_goals

                    # Future conceded (simule aussi)
                    fut_mask = (o_min > t) & (o_min <= t + window)
                    future_conceded = int(o_goals_sim[fut_mask].sum() > 0)

                    is_treated = int(cum_underperf >= threshold)

                    all_diffs.append({
                        'treated': is_treated,
                        'conceded': future_conceded
                    })

        if not all_diffs:
            return None

        df_sim = pd.DataFrame(all_diffs)
        treated = df_sim[df_sim['treated'] == 1]
        control = df_sim[df_sim['treated'] == 0]

        if len(treated) == 0 or len(control) == 0:
            return None

        return treated['conceded'].mean() - control['conceded'].mean()

    @staticmethod
    def print_results(results):
        """Affiche les resultats du null model."""
        r = results
        print("\n" + "=" * 65)
        print("  NULL MODEL BERNOULLI")
        print("=" * 65)
        print(f"  Simulations      : {r['n_simulations']}")
        print(f"  Seuil            : {r['threshold']}")
        print(f"  Fenetre          : {r['window']} min")
        print("-" * 65)
        print(f"  Effet observe    : {r['observed_effect']:+.5f}")
        print(f"  Effet null moyen : {r['null_mean']:+.5f}")
        print(f"  Effet null std   : {r['null_std']:.5f}")
        print(f"  IC 95% null      : [{r['ci_95'][0]:.5f}, {r['ci_95'][1]:.5f}]")
        print(f"  Effet exces      : {r['excess_effect']:+.5f}")
        print("-" * 65)
        print(f"  p-value          : {r['p_value']:.4f}")
        print(f"  Percentile obs.  : {r['percentile_of_observed']:.1f}%")
        print()

        if r['observed_effect'] < r['ci_95'][0]:
            print("  INTERPRETATION : L'effet observe est PLUS NEGATIF que le null.")
            print("  -> Il y a un effet au-dela de la stochasticite pure.")
            print("  -> Les equipes qui underperforment concedent MOINS que prevu")
            print("     meme dans un monde Bernoulli.")
            print("  -> Confounding de force d'equipe probable.")
        elif r['ci_95'][0] <= r['observed_effect'] <= r['ci_95'][1]:
            print("  INTERPRETATION : L'effet observe est DANS le null.")
            print("  -> L'effet est entierement explicable par la stochasticite.")
            print("  -> Pas d'effet reel au-dela du hasard.")
        else:
            print("  INTERPRETATION : L'effet observe est PLUS POSITIF que le null.")
            print("  -> Il y a un vrai effet d'underperformance.")
            print("  -> L'adage 'si tu ne marques pas, tu encaisses' est confirme")
            print("     au-dela de la regression vers la moyenne.")
        print("=" * 65)
