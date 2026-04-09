"""Tests pour le module de nettoyage."""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import pytest

from src.clean import ShotDataCleaner


def make_sample_shots(n=50):
    """Crée un DataFrame de tirs factices pour les tests."""
    rng = np.random.default_rng(42)
    teams = ['Team A', 'Team B']

    rows = []
    for i in range(n):
        team = teams[i % 2]
        rows.append({
            'match_id': 1001,
            'team': team,
            'player': f'Player_{i}',
            'game_minute': sorted(rng.uniform(1, 90, n))[i],
            'xg': rng.uniform(0.02, 0.5),
            'is_goal': int(rng.random() < 0.1),
            'is_penalty': int(rng.random() < 0.05),
            'shot_outcome': 'Goal' if rng.random() < 0.1 else 'Off T',
            'shot_type': 'Open Play',
            'home_team': 'Team A',
            'away_team': 'Team B',
            'score_state': 'drawing',
            'score_diff': 0,
            'source': 'statsbomb',
        })
    return pd.DataFrame(rows)


def test_clean_returns_dataframe():
    cleaner = ShotDataCleaner()
    shots = make_sample_shots()
    result = cleaner.clean(shots, source='statsbomb')
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_clean_removes_bad_xg():
    cleaner = ShotDataCleaner()
    shots = make_sample_shots()
    # Ajouter des xG invalides
    shots.loc[0, 'xg'] = -0.5
    shots.loc[1, 'xg'] = 1.5
    shots.loc[2, 'xg'] = np.nan
    result = cleaner.clean(shots, source='statsbomb')
    assert (result['xg'] >= 0).all()
    assert (result['xg'] <= 1).all()
    assert result['xg'].isna().sum() == 0


def test_quality_report_populated():
    cleaner = ShotDataCleaner()
    shots = make_sample_shots()
    cleaner.clean(shots, source='statsbomb')
    assert 'final_rows' in cleaner.quality_report
    assert cleaner.quality_report['final_rows'] > 0


def test_validate_match_integrity():
    cleaner = ShotDataCleaner()
    shots = make_sample_shots()
    cleaned = cleaner.clean(shots, source='statsbomb')
    issues = cleaner.validate_match_integrity(cleaned)
    # Notre sample a 2 équipes, donc pas de problème attendu
    assert isinstance(issues, pd.DataFrame)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
