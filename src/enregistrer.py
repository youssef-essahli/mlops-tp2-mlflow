import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

exp = mlflow.get_experiment_by_name("tp2-registre")

runs = mlflow.search_runs(
    [exp.experiment_id]
)

configurations = [
    (200, 8),
    (300, 12),
    (500, 16)
]

for n_estimators, max_depth in configurations:

    candidats = runs[
        (runs["params.n_estimators"].astype(str) == str(n_estimators)) &
        (runs["params.max_depth"].astype(str) == str(max_depth))
    ]

    meilleur = candidats.sort_values(
        "metrics.f1",
        ascending=False
    ).iloc[0]

    mv = mlflow.register_model(
        f"runs:/{meilleur['run_id']}/model",
        "fraude"
    )

    print(
        "version",
        mv.version,
        "- configuration",
        n_estimators,
        "/",
        max_depth,
        "- F1",
        round(meilleur["metrics.f1"], 4)
    )