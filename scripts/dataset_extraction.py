import pandas as pd
from pathlib import Path

raw_data_path = Path.cwd().parent / 'ZBIOR_09.05.2021.xlsx'
model_data_path = Path('data') / 'model_data.csv'
text_model_path = Path('data') / 'text_model_data.csv'

raw_df = pd.read_excel(raw_data_path, engine='openpyxl', na_values=['brak'])
raw_df.dropna(how='all', axis=0, inplace=True)

model_data = raw_df.copy()
model_data = model_data[['verb', 'verb - main semantic class', 'verb - tense', 'verb - factive/nonfactive', 'complement - tense', 'T - negation', 'T - type of sentence', 'GOLD <T,H>']]
model_data = model_data[model_data['GOLD <T,H>'].isin(['N', 'E', 'C'])]
model_data['T - negation'] = model_data['T - negation'].astype('bool')

text_model_data = raw_df.copy()
text_model_data = text_model_data[['T PL', 'H PL', 'verb', 'verb - main semantic class', 'verb - tense', 'verb - factive/nonfactive', 'complement - tense', 'T - negation', 'T - type of sentence', 'GOLD <T,H>']]
text_model_data['not_dummified_verb'] = text_model_data['verb']
text_model_data = text_model_data[text_model_data['GOLD <T,H>'].isin(['N', 'E', 'C'])]
text_model_data['T - negation'] = text_model_data['T - negation'].astype('bool')

model_data.to_csv(model_data_path, index=False)
text_model_data.to_csv(text_model_path, index=False)