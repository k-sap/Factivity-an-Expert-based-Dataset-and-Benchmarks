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
 gen_verbs_features, reproduce_model_from_state, BaseNet, gen_base_network, train)

target = 'GOLD <T,H>'

dfs = load_dataframes()

labels_train, labels_test = convert_labels(dfs)

sen_vecs_path = Path('embeddings/sentences_embedded_herbert.pt')
sen_test_vecs_path = Path('embeddings/sentences_test_embedded_herbert.pt')

sen_embedded = torch.load(sen_vecs_path)
sen_test_embedded = torch.load(sen_test_vecs_path)

data = sen_embedded.detach()
test_features = sen_test_embedded.detach()

params = {
    'input_size': 768,
    'hidden_size': 128,
    'output_size': 3,
    'dropout': 0.5
}

lr = 0.001
net, criterion, optimizer = gen_base_network(params, lr)
n_sentences = 1100
batch_size = 50
n_iters = 25

train(net, criterion, optimizer, n_iters, data, labels_train, batch_size, n_sentences, test_features, labels_test)