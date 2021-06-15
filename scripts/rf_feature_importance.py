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
from matplotlib import pyplot as plt

df_paths = {}
dfs = {}

target = 'GOLD <T,H>'

score_path = Path.cwd() / Path('scores') / f'features_comparison_random_forest.csv'

for df_type in ['dev', 'test', 'train']:
    df_paths[df_type] = Path.cwd() / Path('data') / 'split_dummy' / f'{df_type}_data.csv'
    dfs[df_type] = pd.read_csv(df_paths[df_type])
    
    for colname in ['verb', 'verb - main semantic class', 'verb - tense', 'verb - factive/nonfactive', 'complement - tense', 'T - type of sentence', 'GOLD <T,H>']:
        if colname in dfs[df_type].columns:
            dfs[df_type][colname] = dfs[df_type][colname].astype('category')

def dummify(df, target=target):
    columns_to_dummify = [column for column in df if column != target]
    return pd.get_dummies(df, drop_first=True, dummy_na=True, columns=columns_to_dummify)

feature_sets = {
    'factive_only': ['verb - factive/nonfactive'],
    'verb_only': ['verb'],
    'no__factive_and_semantic_class': ['verb', 'verb - tense',
                            'complement - tense', 'T - negation',
                           'T - type of sentence'],
    'no__factive': ['verb', 'verb - tense', 'verb - main semantic class',
                            'complement - tense', 'T - negation',
                           'T - type of sentence'],
    'all': ['verb', 'verb - main semantic class', 'verb - tense',
           'verb - factive/nonfactive', 'complement - tense', 'T - negation',
           'T - type of sentence']
}

def filter_dummy_features(df, features):
    filtered_columns = []
    for column in df.columns:
        if column.split('_')[0] in features:
            filtered_columns.append(column)
    return df[filtered_columns]

def calc_score(dfs, feature_set, model, target=target):
    
    X_train = filter_dummy_features(dfs['train'], feature_set)
    y_train = dfs['train'][target]

    X_test = filter_dummy_features(dfs['test'], feature_set)
    y_test = dfs['test'][target] 

    model.fit(X_train, y_train)

    return {
        'average weighted f1': metrics.f1_score(y_test, model.predict(X_test), average='weighted'),
        'accuracy': metrics.accuracy_score(y_test, model.predict(X_test))
    }

def calc_scores(model):
    accs = []
    f1s = []
    inds = []
    flag = 0
    for feature_set_name, feature_set in feature_sets.items():
        if flag > 9:
            break
        flag += 1

        rt = calc_score(dfs, feature_set, model, target)
        
        accs.append(rt['accuracy'])
        f1s.append(rt['average weighted f1'])
        inds.append(feature_set_name)

    scores = pd.DataFrame(columns=['acc', 'avg weighted f1'], data = np.array([accs, f1s]).T, index=inds)
    
    print(scores)

# scores.to_csv(score_path)


rf = RandomForestClassifier(random_state=10, n_estimators=100)

# calc_scores(lr)
calc_scores(rf)
plt.rcdefaults()
fig, ax = plt.subplots()
indices = np.argsort(rf.feature_importances_)[::-1][:15]
to_plot = pd.DataFrame({
    'feature': list(filter_dummy_features(dfs['train'], feature_sets['all']).columns[indices]),
    'importance': rf.feature_importances_[indices]
})
import seaborn
plt.tight_layout()
seaborn.barplot(y='feature', x='importance', data=to_plot, color='darkblue')
plt.tight_layout()
plt.title('impurity-based feature importances')

plt.savefig('rf_fi.png', dpi=300, bbox_inches="tight")
