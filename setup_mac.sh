#!/bin/bash
# =============================================================
# Setup script — Football Underperformance Analysis
# Exécuter depuis le dossier du projet :
#   chmod +x setup_mac.sh && ./setup_mac.sh
# =============================================================

set -e  # Arrêter en cas d'erreur

echo ""
echo "════════════════════════════════════════════════════"
echo "  Setup — If You Don't Score, You Concede"
echo "════════════════════════════════════════════════════"
echo ""

# 1. Vérifier Python
echo "1. Vérification de Python..."
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "   ✓ $PYTHON_VERSION"
else
    echo "   ✗ Python3 non trouvé !"
    echo "   → Installer via : brew install python3"
    exit 1
fi

# 2. Créer l'environnement virtuel
echo ""
echo "2. Création de l'environnement virtuel..."
if [ -d ".venv" ]; then
    echo "   → .venv existe déjà, skip"
else
    python3 -m venv .venv
    echo "   ✓ .venv créé"
fi

# 3. Activer l'environnement
echo ""
echo "3. Activation de l'environnement..."
source .venv/bin/activate
echo "   ✓ Environnement activé : $(which python)"

# 4. Upgrade pip
echo ""
echo "4. Mise à jour de pip..."
pip install --upgrade pip --quiet
echo "   ✓ pip à jour"

# 5. Installer les dépendances
echo ""
echo "5. Installation des dépendances (peut prendre 2-3 min)..."
pip install -r requirements.txt --quiet
echo "   ✓ Dépendances installées"

# 6. Vérification rapide
echo ""
echo "6. Vérification des imports..."
python3 -c "
import pandas, numpy, scipy, statsmodels, lifelines
import sklearn, matplotlib, seaborn, statsbombpy
import requests, yaml, tqdm, pyarrow
print('   ✓ Tous les packages importés avec succès')
" 2>&1

# 7. Test StatsBomb
echo ""
echo "7. Test de connexion StatsBomb..."
python3 -c "
from statsbombpy import sb
comps = sb.competitions()
print(f'   ✓ StatsBomb OK — {len(comps)} compétitions disponibles')
" 2>&1

# 8. Créer les dossiers de données
echo ""
echo "8. Vérification des dossiers..."
mkdir -p data/{raw/{statsbomb,understat},processed,external}
mkdir -p outputs/{figures,tables,report}
echo "   ✓ Structure de dossiers OK"

# 9. Initialiser Git
echo ""
echo "9. Git..."
if [ -d ".git" ]; then
    echo "   → Repo Git existe déjà"
else
    git init --quiet
    git add .
    git commit -m "Initial commit — project setup" --quiet
    echo "   ✓ Repo Git initialisé + premier commit"
fi

# Résumé
echo ""
echo "════════════════════════════════════════════════════"
echo "  ✓ SETUP TERMINÉ"
echo "════════════════════════════════════════════════════"
echo ""
echo "  Prochaines étapes :"
echo ""
echo "  1. Activer l'environnement (à faire à chaque session) :"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Lancer le premier notebook :"
echo "     cd notebooks"
echo "     jupyter notebook 01_data_exploration.ipynb"
echo ""
echo "  3. Exécuter les cellules une par une et suivre les instructions."
echo ""
