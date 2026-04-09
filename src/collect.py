"""
Module de collecte de données.
Gère StatsBomb Open Data et Understat.
"""

import pandas as pd
import numpy as np
import json
import re
import time
import requests
import logging
from pathlib import Path
from tqdm import tqdm

from statsbombpy import sb

from src.utils import load_config, DATA_RAW, setup_logger

logger = setup_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# STATSBOMB COLLECTOR
# ═══════════════════════════════════════════════════════════════════

class StatsBombCollector:
    """
    Collecte event-level data depuis StatsBomb Open Data.

    Usage :
        collector = StatsBombCollector()
        comps = collector.list_competitions()
        shots = collector.collect_season(competition_id=11, season_id=90)
    """

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir or DATA_RAW / "statsbomb")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Exploration ────────────────────────────────────────────────
    def list_competitions(self):
        """Liste toutes les compétitions disponibles dans StatsBomb Open Data."""
        comps = sb.competitions()
        cols = ['competition_id', 'season_id', 'competition_name',
                'season_name', 'match_available', 'match_updated']
        return comps[[c for c in cols if c in comps.columns]]

    def list_matches(self, competition_id, season_id):
        """Liste les matchs d'une compétition/saison."""
        matches = sb.matches(
            competition_id=competition_id,
            season_id=season_id
        )
        return matches

    # ── Collecte ───────────────────────────────────────────────────
    def collect_season(self, competition_id, season_id):
        """
        Collecte TOUS les événements de tir d'une saison.

        Retourne un DataFrame de tirs avec :
        match_id, team, player, game_minute, xg, is_goal,
        shot_outcome, score_home, score_away, home_team, away_team

        Utilise un cache .parquet par match pour éviter de re-télécharger.
        """
        matches = self.list_matches(competition_id, season_id)
        comp_name = matches['competition'].iloc[0] if 'competition' in matches.columns else str(competition_id)
        season_name = matches['season'].iloc[0] if 'season' in matches.columns else str(season_id)

        logger.info(f"Collecte : {comp_name} {season_name} — {len(matches)} matchs")

        all_shots = []
        errors = []

        for _, match in tqdm(matches.iterrows(), total=len(matches),
                             desc=f"{comp_name} {season_name}"):
            match_id = match['match_id']

            try:
                shots = self._get_match_shots(match_id, match)
                if shots is not None and len(shots) > 0:
                    all_shots.append(shots)
            except Exception as e:
                errors.append({'match_id': match_id, 'error': str(e)})
                logger.warning(f"Erreur match {match_id}: {e}")

        if errors:
            logger.warning(f"{len(errors)} matchs en erreur sur {len(matches)}")

        if not all_shots:
            logger.error("Aucun tir collecté !")
            return pd.DataFrame()

        result = pd.concat(all_shots, ignore_index=True)

        # Ajouter métadonnées de compétition
        result['competition_id'] = competition_id
        result['season_id'] = season_id
        result['competition_name'] = comp_name
        result['season_name'] = season_name

        logger.info(f"✓ {len(result)} tirs collectés sur {len(all_shots)} matchs")

        return result

    def _get_match_shots(self, match_id, match_info):
        """Récupère les tirs d'un match (avec cache)."""
        cache_path = self.cache_dir / f"shots_{match_id}.parquet"

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        # Télécharger les événements
        events = sb.events(match_id=match_id)

        if events is None or len(events) == 0:
            return None

        # Filtrer les tirs
        shots = events[events['type'] == 'Shot'].copy()

        if len(shots) == 0:
            return None

        # Extraire les champs pertinents
        shots = self._parse_shot_fields(shots, match_id, match_info)

        # Sauvegarder en cache
        shots.to_parquet(cache_path, index=False)

        return shots

    def _parse_shot_fields(self, shots, match_id, match_info):
        """Extrait et structure les champs d'un tir StatsBomb."""
        n = len(shots)
        df = pd.DataFrame()

        # Assigner les arrays en premier pour dimensionner le DataFrame
        df['team'] = shots['team'].values
        df['player'] = shots['player'].values
        # Puis le scalaire (broadcast sur n lignes)
        df['match_id'] = match_id

        # Timestamp précis
        df['minute'] = shots['minute'].values
        df['second'] = shots['second'].values if 'second' in shots.columns else 0
        df['game_minute'] = df['minute'] + df['second'] / 60

        # xG — le champ peut avoir différents noms selon la version
        xg_col = None
        for col_name in ['shot_statsbomb_xg', 'shot_xg']:
            if col_name in shots.columns:
                xg_col = col_name
                break

        if xg_col is None:
            # Essayer d'accéder via le dict imbriqué
            logger.warning(f"Match {match_id}: colonne xG non trouvée, tentative alternative")
            df['xg'] = np.nan
        else:
            df['xg'] = shots[xg_col].values

        # Outcome du tir
        df['shot_outcome'] = shots['shot_outcome'].values if 'shot_outcome' in shots.columns else 'Unknown'
        df['is_goal'] = df['shot_outcome'].astype(str).str.contains('Goal', case=False, na=False).astype(int)
        # Exclure les "Own Goal For" qui ne sont pas des vrais goals du tireur
        own_goal_mask = df['shot_outcome'].astype(str).str.contains('Own Goal', case=False, na=False)
        df.loc[own_goal_mask, 'is_goal'] = 0

        # Type de tir (penalty, free kick, etc.)
        df['shot_type'] = shots['shot_type'].values if 'shot_type' in shots.columns else 'Unknown'
        df['is_penalty'] = df['shot_type'].astype(str).str.contains('Penalty', case=False, na=False).astype(int)

        # Body part
        df['body_part'] = shots['shot_body_part'].values if 'shot_body_part' in shots.columns else 'Unknown'

        # Position du tir (coordonnées StatsBomb : terrain 120 × 80)
        if 'location' in shots.columns:
            locations = shots['location'].values
            df['x'] = [loc[0] if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2 else np.nan for loc in locations]
            df['y'] = [loc[1] if isinstance(loc, (list, np.ndarray)) and len(loc) >= 2 else np.nan for loc in locations]
        else:
            df['x'] = np.nan
            df['y'] = np.nan

        # Équipes
        home_team = match_info.get('home_team', 'Unknown')
        away_team = match_info.get('away_team', 'Unknown')
        df['home_team'] = home_team
        df['away_team'] = away_team

        # Score au moment du tir — calcul dynamique
        df = self._compute_running_score(df)

        # Période (1ère mi-temps, 2ème, prolongation)
        df['period'] = shots['period'].values if 'period' in shots.columns else 1

        return df

    def _compute_running_score(self, shots_df):
        """
        Calcule le score courant au moment de chaque tir.
        Le score AVANT le tir (le but éventuel n'est pas encore compté).
        """
        df = shots_df.sort_values('game_minute').copy()

        home = df['home_team'].iloc[0]
        away = df['away_team'].iloc[0]

        # Buts marqués par chaque équipe, dans l'ordre chronologique
        df['is_home_goal'] = ((df['team'] == home) & (df['is_goal'] == 1)).astype(int)
        df['is_away_goal'] = ((df['team'] == away) & (df['is_goal'] == 1)).astype(int)

        # Score AVANT le tir = cumul des buts précédents (shift)
        df['score_home'] = df['is_home_goal'].cumsum().shift(1, fill_value=0)
        df['score_away'] = df['is_away_goal'].cumsum().shift(1, fill_value=0)

        # Score state relatif au tireur
        def _get_score_state(row):
            if row['team'] == home:
                diff = row['score_home'] - row['score_away']
            else:
                diff = row['score_away'] - row['score_home']
            if diff > 0:
                return 'winning'
            elif diff < 0:
                return 'losing'
            return 'drawing'

        df['score_state'] = df.apply(_get_score_state, axis=1)
        df['score_diff'] = df.apply(
            lambda r: (r['score_home'] - r['score_away'])
                      if r['team'] == home
                      else (r['score_away'] - r['score_home']),
            axis=1
        )

        # Cleanup
        df = df.drop(columns=['is_home_goal', 'is_away_goal'])

        return df


# ═══════════════════════════════════════════════════════════════════
# UNDERSTAT COLLECTOR (Phase 2 — à activer plus tard)
# ═══════════════════════════════════════════════════════════════════

class UnderstatCollector:
    """
    Collecte shot-level data depuis Understat (scraping).

    Usage :
        collector = UnderstatCollector()
        shots = collector.collect_season("EPL", "2023")
    """

    BASE_URL = "https://understat.com"

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir or DATA_RAW / "understat")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = load_config()
        self.rate_limit = self.config['data']['understat']['rate_limit_seconds']

    def get_league_matches(self, league, season):
        """
        Récupère la liste des matchs d'une saison.
        league : 'EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1'
        season : '2023' (pour 2023/24)
        """
        cache_path = self.cache_dir / f"matches_{league}_{season}.json"

        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        url = f"{self.BASE_URL}/league/{league}/{season}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        pattern = r"var\s+datesData\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, resp.text)

        if not match:
            logger.warning(f"Pas de données pour {league} {season}")
            return []

        decoded = match.group(1).encode().decode('unicode_escape')
        data = json.loads(decoded)

        with open(cache_path, 'w') as f:
            json.dump(data, f)

        return data

    def get_match_shots(self, match_id):
        """Récupère les tirs d'un match Understat."""
        cache_path = self.cache_dir / f"shots_{match_id}.json"

        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        url = f"{self.BASE_URL}/match/{match_id}"
        resp = requests.get(url, timeout=30)
        time.sleep(self.rate_limit)

        pattern = r"var\s+shotsData\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, resp.text)

        if not match:
            return {"h": [], "a": []}

        decoded = match.group(1).encode().decode('unicode_escape')
        data = json.loads(decoded)

        with open(cache_path, 'w') as f:
            json.dump(data, f)

        return data

    def collect_season(self, league, season):
        """Collecte et structure tous les tirs d'une saison Understat."""
        matches = self.get_league_matches(league, season)
        logger.info(f"Understat {league} {season}: {len(matches)} matchs")

        all_shots = []
        for match_info in tqdm(matches, desc=f"{league} {season}"):
            match_id = match_info['id']
            shots_raw = self.get_match_shots(match_id)

            for side in ['h', 'a']:
                for shot in shots_raw.get(side, []):
                    shot['match_id'] = match_id
                    shot['side'] = side
                    shot['h_team'] = match_info.get('h', {}).get('title', '')
                    shot['a_team'] = match_info.get('a', {}).get('title', '')
                    all_shots.append(shot)

        if not all_shots:
            return pd.DataFrame()

        df = pd.DataFrame(all_shots)

        # Standardiser les types
        df['minute'] = pd.to_numeric(df['minute'], errors='coerce')
        df['xG'] = pd.to_numeric(df['xG'], errors='coerce')
        df['is_goal'] = (df['result'] == 'Goal').astype(int)
        df['game_minute'] = df['minute']

        # Métadonnées
        df['league'] = league
        df['season'] = season
        df['source'] = 'understat'

        logger.info(f"✓ {len(df)} tirs collectés")
        return df
