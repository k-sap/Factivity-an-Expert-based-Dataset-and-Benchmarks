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
 gen_verbs_features, reproduce_model_from_state, BaseNet)

target = 'GOLD <T,H>'

dfs = load_dataframes()

labels_train, labels_test = convert_labels(dfs)

verbs_embedded = gen_verbs_features(dfs['train']['not_dummified_verb'])
verbs_test_embedded = gen_verbs_features(dfs['test']['not_dummified_verb'])

data = verbs_embedded.detach()
test_features = verbs_test_embedded.detach()

params = {
    'input_size': 768,
    'hidden_size': 128,
    'output_size': 3,
    'dropout': 0
}

model_path = Path('models') / 'herbert_verbs_state_mean_acc=0.86_2021-06-04_21_33_59.pt'

reproduce_model_from_state(BaseNet, model_path, test_features.cuda(), labels_test.cuda(), params)