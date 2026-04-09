"""
Module de nettoyage et validation des données de tirs.
"""

import pandas as pd
import numpy as np
import logging

from src.utils import load_config, setup_logger

logger = setup_logger(__name__)


class ShotDataCleaner:
    """
    Nettoie et valide les données de tirs.
    Produit un rapport de qualité automatique.

    Usage :
        cleaner = ShotDataCleaner()
        shots_clean = cleaner.clean(shots_raw, source='statsbomb')
        cleaner.print_quality_report()
    """

    def __init__(self):
        self.config = load_config()
        self.quality_report = {}
        self._issues = []

    def clean(self, shots_df, source='statsbomb'):
        """
        Pipeline de nettoyage complet.

        Étapes :
        1. Standardiser les colonnes
        2. Filtrer les own goals
        3. Flaguer les penalties
        4. Valider les ranges (xG, minutes)
        5. Déduplication
        6. Rapport de qualité

        Parameters
        ----------
        shots_df : pd.DataFrame
            Données brutes de tirs
        source : str
            'statsbomb' ou 'understat'

        Returns
        -------
        pd.DataFrame
            Données nettoyées
        """
        df = shots_df.copy()
        initial_n = len(df)
        logger.info(f"Nettoyage : {initial_n} tirs ({source})")

        # ── 1. Standardiser les colonnes ──
        df = self._standardize_columns(df, source)

        # ── 2. Supprimer les lignes sans xG ──
        n_null_xg = df['xg'].isna().sum()
        df = df.dropna(subset=['xg'])
        if n_null_xg > 0:
            logger.warning(f"  {n_null_xg} tirs sans xG supprimés")

        # ── 3. Valider les ranges ──
        # xG entre 0 et 1
        bad_xg = ((df['xg'] < 0) | (df['xg'] > 1)).sum()
        df = df[(df['xg'] >= 0) & (df['xg'] <= 1)]
        if bad_xg > 0:
            logger.warning(f"  {bad_xg} tirs avec xG hors [0,1]")

        # Minutes entre 0 et 130 (incluant prolongations)
        bad_min = ((df['game_minute'] < 0) | (df['game_minute'] > 130)).sum()
        df = df[(df['game_minute'] >= 0) & (df['game_minute'] <= 130)]
        if bad_min > 0:
            logger.warning(f"  {bad_min} tirs avec minute invalide")

        # ── 4. Flaguer les penalties ──
        if 'is_penalty' not in df.columns:
            df['is_penalty'] = 0
        n_penalties = df['is_penalty'].sum()

        # ── 5. Supprimer les own goals (déjà gérés normalement) ──
        if 'shot_outcome' in df.columns:
            own_goals = df['shot_outcome'].astype(str).str.contains(
                'Own Goal', case=False, na=False
            )
            n_own_goals = own_goals.sum()
            df = df[~own_goals]
        else:
            n_own_goals = 0

        # ── 6. Déduplication ──
        dedup_cols = ['match_id', 'team', 'game_minute', 'xg']
        n_before = len(df)
        df = df.drop_duplicates(subset=dedup_cols)
        n_dupes = n_before - len(df)
        if n_dupes > 0:
            logger.warning(f"  {n_dupes} doublons supprimés")

        # ── 7. Rapport de qualité ──
        self.quality_report = {
            'source': source,
            'initial_rows': initial_n,
            'final_rows': len(df),
            'dropped_total': initial_n - len(df),
            'dropped_pct': round((initial_n - len(df)) / max(initial_n, 1) * 100, 2),
            'null_xg_dropped': n_null_xg,
            'bad_xg_dropped': bad_xg,
            'bad_minute_dropped': bad_min,
            'own_goals_dropped': n_own_goals,
            'duplicates_dropped': n_dupes,
            'penalties_flagged': n_penalties,
            'n_matches': df['match_id'].nunique(),
            'n_teams': df['team'].nunique(),
            'shots_per_match': round(len(df) / max(df['match_id'].nunique(), 1), 1),
            'mean_xg': round(df['xg'].mean(), 4),
            'median_xg': round(df['xg'].median(), 4),
            'conversion_rate': round(df['is_goal'].mean() * 100, 2),
            'penalty_pct': round(df['is_penalty'].mean() * 100, 2),
        }

        logger.info(f"✓ Nettoyage terminé : {len(df)} tirs retenus "
                     f"({self.quality_report['dropped_pct']}% supprimés)")

        return df.reset_index(drop=True)

    def _standardize_columns(self, df, source):
        """Harmonise les noms de colonnes entre StatsBomb et Understat."""
        if source == 'understat':
            rename_map = {
                'xG': 'xg',
                'minute': 'game_minute',
                'h_team': 'home_team',
                'a_team': 'away_team',
                'situation': 'shot_type',
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

            # Équipe du tireur
            if 'team' not in df.columns and 'side' in df.columns:
                df['team'] = df.apply(
                    lambda r: r.get('home_team', '') if r.get('side') == 'h'
                              else r.get('away_team', ''),
                    axis=1
                )

            # Penalty detection pour Understat
            if 'shot_type' in df.columns:
                df['is_penalty'] = df['shot_type'].astype(str).str.contains(
                    'Penalty', case=False, na=False
                ).astype(int)

            df['source'] = 'understat'

        else:
            df['source'] = 'statsbomb'

        return df

    def validate_match_integrity(self, df):
        """
        Vérifie la cohérence interne match par match.
        Retourne un DataFrame des problèmes détectés.
        """
        issues = []

        for match_id, group in df.groupby('match_id'):
            # Exactement 2 équipes
            teams = group['team'].nunique()
            if teams != 2:
                issues.append({
                    'match_id': match_id,
                    'issue': f'{teams} équipes (attendu: 2)',
                    'severity': 'high'
                })

            # Au moins 1 tir par équipe (normalement)
            for team in group['team'].unique():
                team_shots = len(group[group['team'] == team])
                if team_shots == 0:
                    issues.append({
                        'match_id': match_id,
                        'issue': f'{team}: 0 tirs',
                        'severity': 'medium'
                    })

            # Vérification score (si score final disponible)
            total_goals = group['is_goal'].sum()
            if total_goals > 15:
                issues.append({
                    'match_id': match_id,
                    'issue': f'{total_goals} buts (suspect)',
                    'severity': 'medium'
                })

            # Tirs dans l'ordre chronologique
            if not group['game_minute'].is_monotonic_increasing:
                # Pas grave si quelques tirs à la même minute
                pass

        issues_df = pd.DataFrame(issues) if issues else pd.DataFrame(
            columns=['match_id', 'issue', 'severity']
        )

        if len(issues_df) > 0:
            logger.warning(f"⚠ {len(issues_df)} problèmes d'intégrité détectés")
        else:
            logger.info("✓ Intégrité des matchs : OK")

        return issues_df

    def print_quality_report(self):
        """Affiche le rapport de qualité dans la console."""
        r = self.quality_report
        if not r:
            print("Aucun rapport disponible. Exécuter clean() d'abord.")
            return

        print("\n" + "=" * 60)
        print("  RAPPORT DE QUALITÉ — DONNÉES DE TIRS")
        print("=" * 60)
        print(f"  Source              : {r['source']}")
        print(f"  Lignes initiales    : {r['initial_rows']:,}")
        print(f"  Lignes finales      : {r['final_rows']:,}")
        print(f"  Supprimées          : {r['dropped_total']:,} ({r['dropped_pct']}%)")
        print(f"    - xG manquant     : {r['null_xg_dropped']}")
        print(f"    - xG hors range   : {r['bad_xg_dropped']}")
        print(f"    - Minute invalide : {r['bad_minute_dropped']}")
        print(f"    - Own goals       : {r['own_goals_dropped']}")
        print(f"    - Doublons        : {r['duplicates_dropped']}")
        print("-" * 60)
        print(f"  Matchs              : {r['n_matches']}")
        print(f"  Équipes             : {r['n_teams']}")
        print(f"  Tirs / match        : {r['shots_per_match']}")
        print(f"  xG moyen / tir      : {r['mean_xg']}")
        print(f"  xG médian / tir     : {r['median_xg']}")
        print(f"  Taux de conversion  : {r['conversion_rate']}%")
        print(f"  % penalties         : {r['penalty_pct']}%")
        print("=" * 60)

        # Checks de sanité
        print("\n  CHECKS DE SANITÉ :")
        checks = [
            ('Tirs/match entre 15 et 35', 15 <= r['shots_per_match'] <= 35),
            ('xG moyen entre 0.05 et 0.20', 0.05 <= r['mean_xg'] <= 0.20),
            ('Conversion entre 5% et 15%', 5 <= r['conversion_rate'] <= 15),
            ('Penalties < 10%', r['penalty_pct'] < 10),
            ('Moins de 5% de données perdues', r['dropped_pct'] < 5),
        ]
        for label, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {label}")

        print()
