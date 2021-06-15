import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

model_data_path = Path.cwd() / Path('data') / 'text_model_data.csv'

train_data_path = Path.cwd() / Path('data') / 'split_dummy' / 'train_data.csv'
dev_data_path = Path.cwd() / Path('data') / 'split_dummy' / 'dev_data.csv'
test_data_path = Path.cwd() / Path('data') / 'split_dummy' / 'test_data.csv'

model_data = pd.read_csv(model_data_path)
model_data = shuffle(model_data, random_state=0)
for colname in [
    'verb',
    'verb - main semantic class',
    'verb - tense',
    'verb - factive/nonfactive',
    'complement - tense',
    'T - type of sentence',
    'GOLD <T,H>'
    ]:
    model_data[colname] = model_data[colname].astype('category')

data_with_verb = pd.get_dummies(model_data,
 drop_first=True, dummy_na=True,
 columns=['verb', 'verb - main semantic class', 'verb - tense', 'verb - factive/nonfactive', 'complement - tense', 'T - type of sentence'])

X_, X_test, y_, y_test = train_test_split(
data_with_verb.drop(labels='GOLD <T,H>', axis=1), data_with_verb['GOLD <T,H>'], random_state=42,
    train_size=0.55, test_size=0.45, stratify=data_with_verb['GOLD <T,H>'])

X_train, X_dev, y_train, y_dev = train_test_split(
    X_, y_, random_state=42,
    train_size=0.82, test_size=0.18, stratify=y_)

print(
    "data points dimension",\
    np.array((X_train.shape, X_dev.shape, X_test.shape))
)

print(
    "labels count",
    np.array((y_train.value_counts(), y_dev.value_counts(), y_test.value_counts()))
)

X_train.loc[:,'GOLD <T,H>']=y_train
X_dev.loc[:,'GOLD <T,H>']=y_dev
X_test.loc[:,'GOLD <T,H>']=y_test

X_train.to_csv(train_data_path, index=False)
X_dev.to_csv(dev_data_path, index=False)
X_test.to_csv(test_data_path, index=False)