import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri('http://127.0.0.1:5000')

c = MlflowClient()

c.set_registered_model_alias('fraude', 'staging', 2)
c.set_registered_model_alias('fraude', 'production', 3)

c.update_model_version(
    'fraude',
    3,
    description='Validee par l equipe : meilleur F1 sur le jeu de test'
)

for a in ['staging', 'production']:
    v = c.get_model_version_by_alias('fraude', a)
    print(a, '-> version', v.version)