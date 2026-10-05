import sys
import os
import json
import subprocess

import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. Récupération des paramètres depuis la ligne de commande
# ============================================================

n_estimators = int(sys.argv[1])
max_depth = int(sys.argv[2])


# ============================================================
# 2. Configuration MLflow
# ============================================================

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('tp2-registre')

os.makedirs('reports', exist_ok=True)


# ============================================================
# 3. Chargement des données
# ============================================================

df = pd.read_csv('data/transactions.csv')

X = df.drop(columns='fraude')
y = df['fraude']


# ============================================================
# 4. Découpage train / test
# ============================================================

Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# ============================================================
# 5. Exécution MLflow
# ============================================================

with mlflow.start_run(
    run_name=f'rf-{n_estimators}-{max_depth}'
):

    # --------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------

    m = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    ).fit(Xtr, ytr)


    # --------------------------------------------------------
    # Prédictions
    # --------------------------------------------------------

    pred = m.predict(Xte)

    proba = m.predict_proba(Xte)[:, 1]


    # --------------------------------------------------------
    # Paramètres et métriques
    # --------------------------------------------------------

    mlflow.log_params({
        'n_estimators': n_estimators,
        'max_depth': max_depth
    })

    mlflow.log_metrics({
        'f1': f1_score(yte, pred),
        'precision': precision_score(yte, pred),
        'recall': recall_score(yte, pred),
        'roc_auc': roc_auc_score(yte, proba)
    })


    # --------------------------------------------------------
    # Tags de traçabilité
    # --------------------------------------------------------

    commit = subprocess.getoutput(
        'git rev-parse --short HEAD'
    )

    mlflow.set_tags({
        'auteur': 'YOUSSEF ESSAHLI',
        'commit': commit,
        'jeu_de_donnees': 'transactions v1'
    })


    # --------------------------------------------------------
    # Artefact 1 : matrice de confusion
    # --------------------------------------------------------

    ConfusionMatrixDisplay.from_predictions(
        yte,
        pred
    )

    plt.savefig(
        'reports/confusion.png',
        dpi=110,
        bbox_inches='tight'
    )

    plt.close()

    mlflow.log_artifact(
        'reports/confusion.png'
    )


    # --------------------------------------------------------
    # Artefact 2 : importance des variables
    # --------------------------------------------------------

    imp = {
        c: float(round(v, 4))
        for c, v in zip(
            X.columns,
            m.feature_importances_
        )
    }

    json.dump(
        imp,
        open('reports/importances.json', 'w'),
        indent=2
    )

    mlflow.log_artifact(
        'reports/importances.json'
    )


    # --------------------------------------------------------
    # Modèle avec signature et environnement
    # --------------------------------------------------------

    mlflow.sklearn.log_model(
        m,
        name='model',
        input_example=Xte.head(3),
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )


    # --------------------------------------------------------
    # Affichage du F1
    # --------------------------------------------------------

    print(
        'F1 =',
        round(f1_score(yte, pred), 4)
    )