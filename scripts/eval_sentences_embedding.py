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

sen_vecs_path = Path('embeddings/sentences_embedded_herbert.pt')
sen_test_vecs_path = Path('embeddings/sentences_test_embedded_herbert.pt')

sen_embedded = torch.load(sen_vecs_path).cuda()
sen_test_embedded = torch.load(sen_test_vecs_path).cuda()

data = sen_embedded.detach()
test_features = sen_test_embedded.detach()

params = {
    'input_size': 768,
    'hidden_size': 128,
    'output_size': 3,
    'dropout': 0
}

model_path = Path('models') / 'herbert_sentence_state_mean_acc=0.68_balanced_2021-06-14_12_24_36.pt'

reproduce_model_from_state(BaseNet, model_path, test_features.cuda(), labels_test.cuda(), params)