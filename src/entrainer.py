import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

p = yaml.safe_load(
    open('params.yaml', encoding='utf-8')
)['entrainement']

train = pd.read_csv('data/prepare/train.csv')

X = train.drop(columns='fraude')
y = train['fraude']

modele = RandomForestClassifier(
    n_estimators=p['n_estimators'],
    max_depth=p['max_depth'],
    random_state=p['random_state']
).fit(X, y)

os.makedirs('models', exist_ok=True)

joblib.dump(
    modele,
    'models/model.pkl'
)

print('entrainement : modele ecrit dans models/model.pkl')