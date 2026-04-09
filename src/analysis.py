"""
Module d'analyse statistique.
Tests de comparaison, permutations, stratification par score state.
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats
from tqdm import tqdm

from src.utils import load_config, setup_logger

logger = setup_logger(__name__)


class WindowAnalysis:
    """
    Analyse en fenetres temporelles : compare les outcomes
    post-underperformance vs baseline.

    Usage :
        wa = WindowAnalysis()
        results = wa.event_study(timeline, threshold=0.5, window=10)
        stratified = wa.stratified_analysis(timeline, threshold=0.5)
        perm = wa.permutation_test(timeline, threshold=0.5)
    """

    def __init__(self, config=None):
        self.config = config or load_config()
        self.alpha = self.config['analysis']['alpha']

    # ===============================================================
    # BASELINE
    # ===============================================================

    def compute_baseline(self, timeline_df, window=10):
        """Taux de base sur l'ensemble du dataset."""
        col_c = f'future_conceded_{window}min'
        col_x = f'future_xga_{window}min'
        col_s = f'future_opp_shots_{window}min'

        return {
            'p_concede': timeline_df[col_c].mean(),
            'mean_xga': timeline_df[col_x].mean(),
            'mean_opp_shots': timeline_df[col_s].mean(),
            'n': len(timeline_df)
        }

    # ===============================================================
    # EVENT STUDY
    # ===============================================================

    def event_study(self, timeline_df, threshold=0.5, window=10):
        """
        Compare les outcomes quand underperf >= threshold
        vs underperf < threshold.

        Retourne un dict avec les resultats des tests.
        """
        col_c = f'future_conceded_{window}min'
        col_x = f'future_xga_{window}min'
        col_s = f'future_opp_shots_{window}min'

        treated = timeline_df[timeline_df['cum_underperf'] >= threshold]
        control = timeline_df[timeline_df['cum_underperf'] < threshold]

        if len(treated) < 10 or len(control) < 10:
            logger.warning(f"Echantillon trop petit : treated={len(treated)}, control={len(control)}")
            return None

        results = {}

        # --- Test 1 : Probabilite de conceder ---
        p_t = treated[col_c].mean()
        p_c = control[col_c].mean()

        # Z-test de proportions
        count = np.array([treated[col_c].sum(), control[col_c].sum()])
        nobs = np.array([len(treated), len(control)])

        try:
            from statsmodels.stats.proportion import proportions_ztest
            z_stat, p_val = proportions_ztest(count, nobs)
        except Exception:
            z_stat, p_val = np.nan, np.nan

        results['concede'] = {
            'treated_rate': p_t,
            'control_rate': p_c,
            'difference': p_t - p_c,
            'relative_risk': p_t / p_c if p_c > 0 else np.nan,
            'z_stat': z_stat,
            'p_value': p_val,
            'n_treated': len(treated),
            'n_control': len(control),
            'significant': p_val < self.alpha if not np.isnan(p_val) else False
        }

        # --- Test 2 : xGA moyen ---
        t_stat, p_val_x = stats.ttest_ind(
            treated[col_x].dropna(),
            control[col_x].dropna(),
            equal_var=False  # Welch
        )

        # Cohen's d
        pooled_std = np.sqrt(
            (treated[col_x].std()**2 + control[col_x].std()**2) / 2
        )
        d_xga = (treated[col_x].mean() - control[col_x].mean()) / pooled_std if pooled_std > 0 else 0

        results['xga'] = {
            'treated_mean': treated[col_x].mean(),
            'control_mean': control[col_x].mean(),
            'difference': treated[col_x].mean() - control[col_x].mean(),
            't_stat': t_stat,
            'p_value': p_val_x,
            'cohens_d': d_xga,
            'significant': p_val_x < self.alpha
        }

        # --- Test 3 : Nombre de tirs adverses ---
        t_stat_s, p_val_s = stats.ttest_ind(
            treated[col_s].dropna(),
            control[col_s].dropna(),
            equal_var=False
        )

        results['opp_shots'] = {
            'treated_mean': treated[col_s].mean(),
            'control_mean': control[col_s].mean(),
            'difference': treated[col_s].mean() - control[col_s].mean(),
            't_stat': t_stat_s,
            'p_value': p_val_s,
            'significant': p_val_s < self.alpha
        }

        return results

    # ===============================================================
    # ANALYSE STRATIFIEE PAR SCORE STATE
    # ===============================================================

    def stratified_analysis(self, timeline_df, threshold=0.5, window=10):
        """
        Event study stratifie par score state.
        Permet de voir si l'effet est un artefact du score.
        """
        results = {}
        min_n = self.config['analysis']['min_sample_size']

        for state in ['drawing', 'winning', 'losing']:
            subset = timeline_df[timeline_df['score_state'] == state]
            n_treated = len(subset[subset['cum_underperf'] >= threshold])
            n_control = len(subset[subset['cum_underperf'] < threshold])

            if n_treated < min_n or n_control < min_n:
                logger.info(f"  {state}: echantillon insuffisant "
                            f"(treated={n_treated}, control={n_control}, min={min_n})")
                results[state] = {
                    'skipped': True,
                    'reason': f'n_treated={n_treated}, n_control={n_control}',
                    'n_total': len(subset)
                }
                continue

            res = self.event_study(subset, threshold=threshold, window=window)
            if res is not None:
                res['n_total'] = len(subset)
                res['skipped'] = False
                results[state] = res

        return results

    # ===============================================================
    # ANALYSE PAR SEUIL (SENSITIVITY)
    # ===============================================================

    def sensitivity_to_threshold(self, timeline_df, window=10,
                                  thresholds=None):
        """
        Teste plusieurs seuils pour verifier que le resultat
        ne depend pas d'un seuil arbitraire.
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.6, 0.1)

        rows = []
        for thresh in thresholds:
            res = self.event_study(timeline_df, threshold=thresh, window=window)
            if res is None:
                continue

            rows.append({
                'threshold': round(thresh, 2),
                'diff_p_concede': res['concede']['difference'],
                'p_value_concede': res['concede']['p_value'],
                'diff_xga': res['xga']['difference'],
                'p_value_xga': res['xga']['p_value'],
                'cohens_d': res['xga']['cohens_d'],
                'n_treated': res['concede']['n_treated'],
                'n_control': res['concede']['n_control'],
            })

        return pd.DataFrame(rows)

    # ===============================================================
    # TEST DE PERMUTATION
    # ===============================================================

    def permutation_test(self, timeline_df, threshold=0.5, window=10,
                         n_permutations=None, seed=None):
        """
        Test de permutation non-parametrique.
        Shuffle l'assignation traite/controle pour estimer
        la distribution nulle de la difference.
        """
        if n_permutations is None:
            n_permutations = self.config['modeling']['permutation']['n_iterations']
        if seed is None:
            seed = self.config['modeling']['permutation']['random_seed']

        col_c = f'future_conceded_{window}min'

        treated_mask = timeline_df['cum_underperf'] >= threshold
        outcomes = timeline_df[col_c].values

        observed_diff = outcomes[treated_mask].mean() - outcomes[~treated_mask].mean()

        n_treated = treated_mask.sum()
        n_total = len(outcomes)

        rng = np.random.default_rng(seed)
        null_diffs = np.zeros(n_permutations)

        for i in tqdm(range(n_permutations), desc="Permutations",
                      disable=n_permutations < 1000):
            perm_idx = rng.choice(n_total, size=n_treated, replace=False)
            perm_mask = np.zeros(n_total, dtype=bool)
            perm_mask[perm_idx] = True

            null_diffs[i] = outcomes[perm_mask].mean() - outcomes[~perm_mask].mean()

        # P-value bilaterale
        p_value = (np.abs(null_diffs) >= np.abs(observed_diff)).mean()

        return {
            'observed_diff': observed_diff,
            'p_value': p_value,
            'null_mean': null_diffs.mean(),
            'null_std': null_diffs.std(),
            'ci_95_null': (np.percentile(null_diffs, 2.5),
                           np.percentile(null_diffs, 97.5)),
            'n_permutations': n_permutations,
            'threshold': threshold,
            'n_treated': n_treated,
            'n_control': n_total - n_treated,
            'null_distribution': null_diffs
        }

    # ===============================================================
    # CORRECTION TESTS MULTIPLES
    # ===============================================================

    @staticmethod
    def correct_multiple_tests(p_values, method='fdr_bh'):
        """
        Correction de Benjamini-Hochberg pour tests multiples.
        """
        from statsmodels.stats.multitest import multipletests
        p_arr = np.array(p_values)

        # Gerer les NaN
        valid = ~np.isnan(p_arr)
        corrected = np.full_like(p_arr, np.nan)
        reject = np.full(len(p_arr), False)

        if valid.sum() > 0:
            rej, corr, _, _ = multipletests(p_arr[valid], alpha=0.05, method=method)
            corrected[valid] = corr
            reject[valid] = rej

        return corrected, reject

    # ===============================================================
    # AFFICHAGE
    # ===============================================================

    @staticmethod
    def print_event_study(results, label=""):
        """Affiche les resultats d'un event study."""
        if results is None:
            print(f"  {label}: Pas de resultats")
            return

        if results.get('skipped'):
            print(f"  {label}: Skip ({results.get('reason', '')})")
            return

        print(f"\n{'=' * 65}")
        if label:
            print(f"  {label}")
            print(f"{'=' * 65}")

        # Concede
        c = results['concede']
        sig = '*' if c['significant'] else ''
        print(f"\n  P(conceder dans 10 min) :")
        print(f"    Traite   : {c['treated_rate']:.4f}  (n={c['n_treated']})")
        print(f"    Controle : {c['control_rate']:.4f}  (n={c['n_control']})")
        print(f"    Diff     : {c['difference']:+.4f}  "
              f"(RR={c['relative_risk']:.3f})")
        print(f"    z={c['z_stat']:.3f}, p={c['p_value']:.4f} {sig}")

        # xGA
        x = results['xga']
        sig = '*' if x['significant'] else ''
        print(f"\n  xGA moyen (10 min) :")
        print(f"    Traite   : {x['treated_mean']:.4f}")
        print(f"    Controle : {x['control_mean']:.4f}")
        print(f"    Diff     : {x['difference']:+.4f}  "
              f"(Cohen's d={x['cohens_d']:.3f})")
        print(f"    t={x['t_stat']:.3f}, p={x['p_value']:.4f} {sig}")

        # Shots
        s = results['opp_shots']
        sig = '*' if s['significant'] else ''
        print(f"\n  Tirs adverses (10 min) :")
        print(f"    Traite   : {s['treated_mean']:.3f}")
        print(f"    Controle : {s['control_mean']:.3f}")
        print(f"    Diff     : {s['difference']:+.3f}")
        print(f"    t={s['t_stat']:.3f}, p={s['p_value']:.4f} {sig}")

    @staticmethod
    def print_permutation(perm_results):
        """Affiche les resultats d'un test de permutation."""
        r = perm_results
        print(f"\n{'=' * 55}")
        print(f"  TEST DE PERMUTATION (n={r['n_permutations']})")
        print(f"{'=' * 55}")
        print(f"  Seuil            : {r['threshold']}")
        print(f"  Diff observee    : {r['observed_diff']:+.5f}")
        print(f"  Null mean        : {r['null_mean']:+.5f}")
        print(f"  Null std         : {r['null_std']:.5f}")
        print(f"  IC 95% null      : [{r['ci_95_null'][0]:.5f}, {r['ci_95_null'][1]:.5f}]")
        print(f"  p-value          : {r['p_value']:.4f}")
        sig = "OUI" if r['p_value'] < 0.05 else "NON"
        print(f"  Significatif     : {sig}")
        print()
