"""
Module de feature engineering.
Construit les timelines minute-par-minute et les episodes d'underperformance.
"""

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

from src.utils import load_config, setup_logger

logger = setup_logger(__name__)


class UnderperformanceFeatures:
    """
    Construit les features d'underperformance offensive
    et de vulnerabilite defensive.

    Usage :
        feat = UnderperformanceFeatures()
        timeline = feat.build_all_timelines(shots_clean)
        episodes = feat.create_underperformance_episodes(timeline)
    """

    def __init__(self, config=None):
        self.config = config or load_config()
        self.rolling_windows = self.config['analysis']['rolling_windows']
        self.future_windows = self.config['analysis']['future_windows']
        self.thresholds = self.config['analysis']['underperf_thresholds']
        self.exclude_last = self.config['analysis']['exclude_last_minutes']

    # ===============================================================
    # TIMELINES
    # ===============================================================

    def build_all_timelines(self, shots_df):
        """
        Construit les timelines pour tous les matchs du dataset.

        Parameters
        ----------
        shots_df : pd.DataFrame
            Dataset de tirs nettoye (sortie de ShotDataCleaner)

        Returns
        -------
        pd.DataFrame
            Timeline minute-par-minute avec features et outcomes
        """
        match_ids = shots_df['match_id'].unique()
        logger.info(f"Construction des timelines pour {len(match_ids)} matchs")

        all_timelines = []
        errors = []

        for match_id in tqdm(match_ids, desc="Timelines"):
            try:
                match_shots = shots_df[shots_df['match_id'] == match_id]
                tl = self.build_match_timeline(match_shots)
                if tl is not None and len(tl) > 0:
                    all_timelines.append(tl)
            except Exception as e:
                errors.append({'match_id': match_id, 'error': str(e)})
                logger.warning(f"Erreur timeline match {match_id}: {e}")

        if errors:
            logger.warning(f"{len(errors)} matchs en erreur")

        if not all_timelines:
            logger.error("Aucune timeline construite !")
            return pd.DataFrame()

        result = pd.concat(all_timelines, ignore_index=True)
        logger.info(f"Timelines OK : {len(result)} observations "
                     f"sur {len(all_timelines)} matchs")

        return result

    def build_match_timeline(self, match_shots):
        """
        Construit une timeline minute-par-minute pour un match,
        du point de vue de chaque equipe.

        Pour chaque minute t et chaque equipe :
        - Features PASSEES : xG cumule, buts, underperformance, rolling stats
        - Features FUTURES : xGA, buts encaisses dans les fenetres [t, t+delta]

        Returns
        -------
        pd.DataFrame
            Une ligne par (minute, equipe)
        """
        teams = match_shots['team'].unique()
        if len(teams) != 2:
            return None

        match_id = match_shots['match_id'].iloc[0]
        max_minute = int(min(match_shots['game_minute'].max() + 1, 95))

        # Pre-indexer les tirs par equipe
        team_data = {}
        for team in teams:
            team_shots = match_shots[match_shots['team'] == team].sort_values('game_minute')
            opponent = [t for t in teams if t != team][0]
            opp_shots = match_shots[match_shots['team'] == opponent].sort_values('game_minute')
            team_data[team] = {
                'shots': team_shots,
                'opp_shots': opp_shots,
                'opponent': opponent
            }

        rows = []

        for team in teams:
            td = team_data[team]
            team_shots = td['shots']
            opp_shots = td['opp_shots']
            opponent = td['opponent']

            # Arrays pour vectoriser
            t_minutes = team_shots['game_minute'].values
            t_xg = team_shots['xg'].values
            t_goals = team_shots['is_goal'].values

            o_minutes = opp_shots['game_minute'].values
            o_xg = opp_shots['xg'].values
            o_goals = opp_shots['is_goal'].values

            home_team = match_shots['home_team'].iloc[0]

            for t in range(1, max_minute + 1):
                row = self._compute_minute_features(
                    t, match_id, team, opponent, home_team,
                    t_minutes, t_xg, t_goals,
                    o_minutes, o_xg, o_goals,
                    max_minute
                )
                rows.append(row)

        return pd.DataFrame(rows)

    def _compute_minute_features(self, t, match_id, team, opponent,
                                  home_team, t_min, t_xg, t_goals,
                                  o_min, o_xg, o_goals, max_minute):
        """Calcule toutes les features pour une minute donnee."""

        # -- Masques temporels --
        team_past = t_min <= t
        opp_past = o_min <= t

        # -- Features cumulatives (PASSE) --
        cum_xg = t_xg[team_past].sum()
        cum_goals = t_goals[team_past].sum()
        cum_underperf = cum_xg - cum_goals
        cum_shots = team_past.sum()

        # Score state
        opp_goals_at_t = o_goals[opp_past].sum()
        score_diff = cum_goals - opp_goals_at_t

        if score_diff > 0:
            score_state = 'winning'
        elif score_diff < 0:
            score_state = 'losing'
        else:
            score_state = 'drawing'

        # -- Tirs depuis le dernier but (sequence sans but) --
        goal_times = t_min[(t_goals == 1) & team_past]
        if len(goal_times) > 0:
            last_goal_time = goal_times.max()
        else:
            last_goal_time = 0

        since_goal_mask = (t_min > last_goal_time) & team_past
        shots_since_last_goal = since_goal_mask.sum()
        xg_since_last_goal = t_xg[since_goal_mask].sum()

        # -- Rolling features (fenetres passees) --
        rolling = {}
        for w in self.rolling_windows:
            w_mask = (t_min > t - w) & (t_min <= t)
            rolling[f'rolling_xg_{w}min'] = t_xg[w_mask].sum()
            rolling[f'rolling_goals_{w}min'] = t_goals[w_mask].sum()
            rolling[f'rolling_shots_{w}min'] = int(w_mask.sum())
            rolling[f'rolling_underperf_{w}min'] = (
                t_xg[w_mask].sum() - t_goals[w_mask].sum()
            )

        # -- Variables REPONSE (fenetres futures) --
        response = {}
        for w in self.future_windows:
            fut_mask = (o_min > t) & (o_min <= t + w)
            response[f'future_xga_{w}min'] = o_xg[fut_mask].sum()
            response[f'future_goals_against_{w}min'] = int(o_goals[fut_mask].sum())
            response[f'future_opp_shots_{w}min'] = int(fut_mask.sum())
            response[f'future_conceded_{w}min'] = int(o_goals[fut_mask].sum() > 0)

            # Fenetre complete ? (pour filtrer les bords)
            response[f'future_window_complete_{w}min'] = int(t + w <= max_minute)

        # -- Assemblage --
        row = {
            'match_id': match_id,
            'team': team,
            'opponent': opponent,
            'is_home': int(team == home_team),
            'minute': t,
            'cum_xg': round(cum_xg, 6),
            'cum_goals': int(cum_goals),
            'cum_underperf': round(cum_underperf, 6),
            'cum_shots': int(cum_shots),
            'score_diff': int(score_diff),
            'score_state': score_state,
            'shots_since_last_goal': int(shots_since_last_goal),
            'xg_since_last_goal': round(xg_since_last_goal, 6),
            **rolling,
            **response,
        }

        return row

    # ===============================================================
    # EPISODES D'UNDERPERFORMANCE
    # ===============================================================

    def create_underperformance_episodes(self, timeline_df):
        """
        Identifie les episodes d'underperformance :
        le premier moment ou cum_underperf depasse chaque seuil.

        On utilise l'Option A (premier franchissement par match/equipe),
        qui est plus conservatrice et evite les dependances entre episodes.

        Returns
        -------
        pd.DataFrame
            Un episode = un franchissement de seuil avec contexte et outcomes
        """
        episodes = []

        for (match_id, team), group in timeline_df.groupby(['match_id', 'team']):
            group = group.sort_values('minute')

            for threshold in self.thresholds:
                above = group[group['cum_underperf'] >= threshold]

                if len(above) == 0:
                    continue

                # Premier franchissement uniquement (Option A)
                trigger = above.iloc[0]

                episode = {
                    'match_id': match_id,
                    'team': team,
                    'opponent': trigger['opponent'],
                    'threshold': threshold,
                    'trigger_minute': trigger['minute'],
                    'cum_underperf_at_trigger': trigger['cum_underperf'],
                    'cum_xg_at_trigger': trigger['cum_xg'],
                    'cum_goals_at_trigger': trigger['cum_goals'],
                    'cum_shots_at_trigger': trigger['cum_shots'],
                    'score_state_at_trigger': trigger['score_state'],
                    'score_diff_at_trigger': trigger['score_diff'],
                    'shots_since_last_goal': trigger['shots_since_last_goal'],
                    'xg_since_last_goal': trigger['xg_since_last_goal'],
                    'is_home': trigger['is_home'],
                }

                # Outcomes par fenetre
                for w in self.future_windows:
                    for col in [f'future_xga_{w}min',
                                f'future_goals_against_{w}min',
                                f'future_opp_shots_{w}min',
                                f'future_conceded_{w}min',
                                f'future_window_complete_{w}min']:
                        if col in trigger.index:
                            episode[col] = trigger[col]

                episodes.append(episode)

        result = pd.DataFrame(episodes)

        if len(result) > 0:
            logger.info(f"Episodes OK : {len(result)} episodes identifies "
                         f"({len(result[result['threshold']==self.thresholds[0]])} "
                         f"au seuil {self.thresholds[0]})")
        else:
            logger.warning("Aucun episode d'underperformance trouve")

        return result

    # ===============================================================
    # FILTRAGE & UTILITAIRES
    # ===============================================================

    def filter_complete_windows(self, timeline_df, window=10):
        """
        Filtre pour ne garder que les observations ou la fenetre
        future est complete (pas de troncature en fin de match).
        """
        col = f'future_window_complete_{window}min'
        if col in timeline_df.columns:
            filtered = timeline_df[timeline_df[col] == 1].copy()
            dropped = len(timeline_df) - len(filtered)
            logger.info(f"Filtrage fenetre {window}min : {dropped} obs. "
                         f"tronquees supprimees ({dropped/len(timeline_df)*100:.1f}%)")
            return filtered
        return timeline_df

    def summary_stats(self, timeline_df):
        """Statistiques descriptives rapides des timelines."""
        stats = {
            'n_observations': len(timeline_df),
            'n_matches': timeline_df['match_id'].nunique(),
            'n_teams': timeline_df['team'].nunique(),
            'minutes_per_match': timeline_df.groupby('match_id')['minute'].max().mean(),
        }

        # Stats d'underperformance
        stats['cum_underperf_mean'] = timeline_df['cum_underperf'].mean()
        stats['cum_underperf_std'] = timeline_df['cum_underperf'].std()
        stats['cum_underperf_max'] = timeline_df['cum_underperf'].max()
        stats['pct_underperforming'] = (timeline_df['cum_underperf'] > 0).mean() * 100

        # Score states
        state_counts = timeline_df['score_state'].value_counts(normalize=True) * 100
        for state in ['drawing', 'winning', 'losing']:
            stats[f'pct_{state}'] = state_counts.get(state, 0)

        return stats

    def print_summary(self, timeline_df):
        """Affiche un resume lisible."""
        s = self.summary_stats(timeline_df)

        print("\n" + "=" * 60)
        print("  RESUME DES TIMELINES")
        print("=" * 60)
        print(f"  Observations       : {s['n_observations']:,}")
        print(f"  Matchs             : {s['n_matches']}")
        print(f"  Equipes            : {s['n_teams']}")
        print(f"  Min/match (moy.)   : {s['minutes_per_match']:.0f}")
        print("-" * 60)
        print(f"  Underperf. moyen   : {s['cum_underperf_mean']:.3f}")
        print(f"  Underperf. max     : {s['cum_underperf_max']:.3f}")
        print(f"  % obs. underperf>0 : {s['pct_underperforming']:.1f}%")
        print("-" * 60)
        print(f"  % drawing          : {s['pct_drawing']:.1f}%")
        print(f"  % winning          : {s['pct_winning']:.1f}%")
        print(f"  % losing           : {s['pct_losing']:.1f}%")
        print("=" * 60)
        print()
