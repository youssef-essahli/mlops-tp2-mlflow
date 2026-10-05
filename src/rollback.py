import mlflow
import pandas as pd
from mlflow import MlflowClient

mlflow.set_tracking_uri('http://127.0.0.1:5000')

c = MlflowClient()

X = pd.DataFrame([{
    'montant': 8500.0,
    'anciennete': 3,
    'freq_24h': 9,
    'nuit': 1
}])


def servi():
    v = c.get_model_version_by_alias(
        'fraude',
        'production'
    ).version

    m = mlflow.pyfunc.load_model(
        'models:/fraude@production'
    )

    print(
        'alias production -> version',
        v,
        '| run',
        str(m.metadata.run_id)[:8],
        '| prediction',
        m.predict(X)
    )


print('--- etat initial')
servi()

c.set_registered_model_alias(
    'fraude',
    'production',
    2
)

print('--- apres promotion de la v2')
servi()

c.set_registered_model_alias(
    'fraude',
    'production',
    3
)

print('--- apres retour arriere vers la v3')
servi()