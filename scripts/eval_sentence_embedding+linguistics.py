import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import metrics
import datetime

import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from train_eval_utils import (load_dataframes, convert_labels,
 gen_verbs_features, reproduce_model_from_state, BaseNet, linguistics_features)

target = 'GOLD <T,H>'

dfs = load_dataframes()

labels_train, labels_test = convert_labels(dfs)

sen_vecs_path = Path('embeddings/sentences_embedded_herbert.pt')
sen_test_vecs_path = Path('embeddings/sentences_test_embedded_herbert.pt')

sen_embedded = torch.load(sen_vecs_path).cuda()
sen_test_embedded = torch.load(sen_test_vecs_path).cuda()



linguistic_train = torch.from_numpy(dfs['train'][linguistics_features].to_numpy().astype(int)).cuda()
linguistic_test = torch.from_numpy(dfs['test'][linguistics_features].to_numpy().astype(int)).cuda()

data = torch.cat((sen_embedded.detach(), linguistic_train), dim=1)
test_features = torch.cat((sen_test_embedded.detach(), linguistic_test), dim=1)

params = {
    'input_size': 1117,
    'hidden_size': 128,
    'output_size': 3,
    'dropout': 0
}

model_path = Path('models') / 'herbert_sentence+linguistics_state_mean_acc=0.83.43_balanced_2021-06-14_12_19_35.pt'

reproduce_model_from_state(BaseNet, model_path, test_features.cuda(), labels_test.cuda(), params)