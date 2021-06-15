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


def load_dataframes():
    df_paths = {}
    dfs = {}

    target = 'GOLD <T,H>'

    for df_type in ['dev', 'test', 'train']:
        df_paths[df_type] = Path('data') / 'split_dummy' / f'{df_type}_data.csv'
        dfs[df_type] = pd.read_csv(df_paths[df_type])

        for colname in ['verb', 'verb - main semantic class', 'verb - tense', 'verb - factive/nonfactive', 'complement - tense', 'T - type of sentence', 'GOLD <T,H>']:
            if colname in dfs[df_type].columns:
                dfs[df_type][colname] = dfs[df_type][colname].astype('category')
    return dfs

def convert_labels(dfs):
    labels_train = torch.tensor(dfs['train']['GOLD <T,H>'].cat.codes.to_numpy()).to('cuda')
    labels_test = torch.tensor(dfs['test']['GOLD <T,H>'].cat.codes.to_numpy()).to('cuda')
    return labels_train, labels_test

def gen_verb_cleaner():
    return lambda text: re.sub(r'\[[^[]+\]',
        r'',
        text.strip('_')).strip(' ')

def vectorize(sentences, aggregate=torch.mean):
    """ vectgorize string to herbert embeding and aggregate to 768 length vector """
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
    model = AutoModel.from_pretrained("allegro/herbert-klej-cased-v1")
    tokenized = tokenizer(sentences, return_tensors='pt', padding=True, truncation=False, max_length=512)
    embedded = model(**tokenized)
    return aggregate(embedded.last_hidden_state, dim=1).cuda()

def gen_verbs_features(verbs, cut_on_że=False, aggregate=torch.mean):
    """ vectorize verb feature with regex preprocessing """
    cleaned_verbs = list(verbs.apply(gen_verb_cleaner()))
    if cut_on_że:
        return vectorize([verb.split(', że')[0] for verb in cleaned_verbs], aggregate)
    else:
        return vectorize(cleaned_verbs)


class BaseNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(BaseNet, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x, train=True):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if train:
            x = self.dropout(x)
        output = self.fc3(x)
        return F.log_softmax(output, dim=0)


def gen_base_network(params, lr):
    net = BaseNet(**params)
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    return net, criterion, optimizer

def train(net, criterion, optimizer, n_iters, data, data_labels, batch_size, n_sentences, test_features, test_labels):
    scores = []
    for epoch in range(n_iters):  # loop over the dataset multiple times

        running_loss = 0.0
        for ind in range(0, n_sentences, batch_size):
            inputs = torch.tensor(data[ind:ind+batch_size,]).cuda()
            labels = data_labels[ind:ind+batch_size,].long().cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward(retain_graph=False)
            optimizer.step()

            loss_ = loss.item()
            
        if epoch % 2 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, ind + 1, loss_), end=' | \n')
            running_loss = 0.0
        rt = calc_metrics(net, test_features, test_labels)
        scores.append((epoch, rt))
        
        if rt['acc'] > 0.905:
            break
            print('threshold')

    print('Finished')

def calc_metrics(net, vectorized_test, labels_test):
    n_obs = vectorized_test.shape[0]
    predictions = torch.Tensor().new_ones(n_obs).cuda()
    
    result = {}
    
    for ind in range(n_obs):
        probs = net.forward(torch.tensor(vectorized_test[ind]).cuda(), train=False)
        predictions[ind] = torch.argmax(probs)
        
    for label_index, label in enumerate(['C', 'E', 'N']):
        inds_per_class = labels_test == label_index
        vectorized_test_per_class = vectorized_test[inds_per_class, :]
        n_obs_per_class = vectorized_test_per_class.shape[0]
        predictions_per_class = torch.Tensor().new_ones(n_obs_per_class).cuda()
        
        for ind in range(n_obs_per_class):
            probs = net.forward(torch.tensor(vectorized_test_per_class[ind]).cuda(), train=False)
            predictions_per_class[ind] = torch.argmax(probs)
            
        result['acc_' + label] = float(torch.sum(predictions_per_class == labels_test[inds_per_class])
                                       / n_obs_per_class)
        
    
    result.update({
        'acc': float(torch.sum(predictions == labels_test[:n_obs]) / predictions.shape[0]), 
        'f1': metrics.f1_score(labels_test[:n_obs].cpu(), predictions.cpu(), average='weighted')
    })
    print(result)
    return result

def reproduce_model_from_state(net_class, model_path, vectorized_test, labels_test, params):
    net = net_class(**params)
    net.cuda()
    net.load_state_dict(torch.load(model_path))
    return calc_metrics(net, vectorized_test, labels_test)


linguistics_features = ['T - negation',
 'verb_PODEJRZEWAĆ, że_',
 'verb_POMYŚLEĆ, że_',
 'verb_WYDAJE się komuś, że_',
 'verb_[e] CZUĆ, że_',
 'verb_[e] czuć, że_',
 'verb_[e] poczuć, że_',
 'verb_[e] spostrzec, że_',
 'verb_[e] widać, że_',
 'verb_[e] widzieć, że_',
 'verb_[e] zauważać, że_',
 'verb_[e] zauważyć, że_',
 'verb_[e] zobaczyć, że_',
 'verb_[elipsa]',
 'verb_[m] ustalić, że_',
 'verb_[m] wskazywać, że_',
 'verb_[m] zauważyć, że_',
 'verb_[p] czuć, że_',
 'verb_[p] dostrzec, że_',
 'verb_[p] obserwować, że_',
 'verb_[p] poczuć, że_',
 'verb_[p] słychać, że_[przyroda]',
 'verb_[p] słyszeć, że_',
 'verb_[p] widać, że_',
 'verb_[p] widzieć, że_',
 'verb_[p] zaobserwować, że_',
 'verb_[p] zauważyć, że_',
 'verb_[p] zobaczyć, że_',
 'verb_alarmować, że_',
 'verb_argumentować, że_',
 'verb_bać się, że_',
 'verb_bywać, że_',
 'verb_być dumnym, że_',
 'verb_być pewnym, że_',
 'verb_być przekonanym, że_',
 'verb_być zdania, że_',
 'verb_być zgodnym, że_',
 'verb_być świadomym, że_',
 'verb_cieszyć się, że_',
 'verb_coś pokazało, że_',
 'verb_coś sygnalizuje, że_',
 'verb_czepiać się, że_',
 'verb_czytać, że_',
 'verb_dać do zrozumienia, że_',
 'verb_dać poznać, że_',
 'verb_dać słowo honoru, że_',
 'verb_dać znać, że_',
 'verb_decydować, że_',
 'verb_deklarować, że_',
 'verb_dodawać, że_',
 'verb_dodać, że_',
 'verb_domyślam(y) się, że_',
 'verb_domyślać się, że_',
 'verb_domyślić się, że_',
 'verb_dopowiedzieć, że_',
 'verb_dowiadywać się, że_',
 'verb_dowiedzieć się, że_',
 'verb_dowieść, że_',
 'verb_dowodzić, że_',
 'verb_dziać się, że_',
 'verb_dziwić się, że_',
 'verb_grozić, że_',
 'verb_grzmieć, że_',
 'verb_gwarantować, że_',
 'verb_głosić, że_',
 'verb_informować, że_',
 'verb_jęczeć, że_',
 'verb_kogoś martwi [coś], że_',
 'verb_krakać, że_',
 'verb_ktoś nie ma wątpliwości, że_',
 'verb_ktoś pokazał, że_',
 'verb_ktoś pokazał, że_2',
 'verb_ktoś pokazuje, że_1',
 'verb_ktoś sygnalizuje, że_',
 'verb_lamentować, że_',
 'verb_liczyć się z czymś, że_',
 'verb_liczyć, że_',
 'verb_martwić się, że_',
 'verb_meldować, że_',
 'verb_miarkować, że_',
 'verb_mieć nadzieję, że_',
 'verb_mieć pewność, że_',
 'verb_mieć wrażenie, że_',
 'verb_mieć świadomość, że_',
 'verb_mniemać, że_',
 'verb_myśleć, że_',
 'verb_móc się domyślać, że_',
 'verb_mówić, że_',
 'verb_nadmienić, że_',
 'verb_napisać, że_',
 'verb_napomknąć, że_',
 'verb_narzekać, że_',
 'verb_nie da się ukryć, że_',
 'verb_nie kryć, że_',
 'verb_nie ukrywać, że_',
 'verb_nie ulega wątpliwości, że_',
 'verb_niepokoić się, że_',
 'verb_obawiam się, że_',
 'verb_obawiać się, że_',
 'verb_obić się o uszy, że_',
 'verb_obliczać, że_',
 'verb_obliczyć, że_',
 'verb_oceniać, że_',
 'verb_ocenić, że_',
 'verb_oczekiwać, że_',
 'verb_odczuwać, że_',
 'verb_odkrywać, że_',
 'verb_odkryć, że_',
 'verb_odnieść wrażenie, że_',
 'verb_odnosić wrażenie, że_',
 'verb_odpisać, że_',
 'verb_odpowiadać, że_',
 'verb_odpowiedzieć, że_',
 'verb_ogłosić, że_',
 'verb_okazać się, że_',
 'verb_okazywać się, że_',
 'verb_olśniło kogoś, że_',
 'verb_opowiadać, że_',
 'verb_orientować się, że_',
 'verb_orzec, że_',
 'verb_oszacować, że_',
 'verb_oznajmić, że_',
 'verb_oświadczać, że_',
 'verb_oświadczyć, że_',
 'verb_pamiętać, że_',
 'verb_pisać, że_',
 'verb_planować, że_',
 'verb_podawać, że_',
 'verb_podać, że_',
 'verb_podejrzewać, że_',
 'verb_podkreślać, że_',
 'verb_podkreślić, że_',
 'verb_podpisywać, że_',
 'verb_podpowiadać, że_',
 'verb_poinformować, że_',
 'verb_pojmować, że_',
 'verb_pokazać, że_',
 'verb_pokazać, że_1',
 'verb_policzyć, że_',
 'verb_pomyśleć, że_',
 'verb_postanowić, że_',
 'verb_potwierdzać, że_',
 'verb_potwierdzić, że_',
 'verb_powiadać, że_',
 'verb_powiedzieć, że_',
 'verb_powtarzać, że_',
 'verb_poznać, że_',
 'verb_przeczuwać, że_',
 'verb_przeczytać, że_',
 'verb_przekazywać, że_',
 'verb_przekonać [kogoś], że_',
 'verb_przekonać się, że_',
 'verb_przekonywać [kogoś], że_',
 'verb_przekonywać się, że_',
 'verb_przekonywać, że_',
 'verb_przemówiło do kogoś, że_',
 'verb_przeoczyć, że_',
 'verb_przerazić się, że_',
 'verb_przewidywać, że_',
 'verb_przewidzieć, że_',
 'verb_przyjmować, że_',
 'verb_przyjąć, że_',
 'verb_przypominać komuś, że_',
 'verb_przypominać sobie, że_',
 'verb_przypomnieć komuś, że_',
 'verb_przypomnieć sobie, że_',
 'verb_przypuszczać, że_',
 'verb_przyrzekać, że_',
 'verb_przysiąc, że_',
 'verb_przysięgać, że_',
 'verb_przyznawać się, że_',
 'verb_przyznawać, że_',
 'verb_przyznać się, że_',
 'verb_przyznać, że_',
 'verb_rechotać, że_',
 'verb_rozumieć coś w jakis sposób, że_',
 'verb_rozumieć coś w jakiś sposób, że_',
 'verb_rozumieć, że_',
 'verb_skłamać, że_',
 'verb_spodziewać się, że_',
 'verb_sprawiać wrażenie, że_',
 'verb_sprawiać, że_',
 'verb_sprawić, że_',
 'verb_stać, że_',
 'verb_stwierdzam, że_',
 'verb_stwierdzać, że_',
 'verb_stwierdzić, że_',
 'verb_sugerować, że_',
 'verb_szacować, że_',
 'verb_szepnąć, że_',
 'verb_szeptać, że_',
 'verb_sądzić, że_',
 'verb_słyszeć, że_[mowa]',
 'verb_twierdzić, że_',
 'verb_tłuc, że_',
 'verb_tłumaczyć się, że_',
 'verb_tłumaczyć, że_',
 'verb_uczyć, że_',
 'verb_udawać, że_',
 'verb_udać, że_',
 'verb_udokumentować, że_',
 'verb_udowadniać, że_',
 'verb_udowodnić, że_',
 'verb_ukazać, że_',
 'verb_umówić się, że_',
 'verb_unaocznić, że_',
 'verb_upewnić się, że_',
 'verb_upierać się, że_',
 'verb_uprzedzić, że_',
 'verb_uprzytomnić sobie, że_',
 'verb_uroić sobie, że_',
 'verb_uspokajać, że_',
 'verb_usprawiedliwiać się, że_',
 'verb_ustalać sobie, że_',
 'verb_ustalić [na podstawie czegoś], że_',
 'verb_usłyszeć, że_[mowa] ',
 'verb_utrzymywać, że_',
 'verb_uważać, że_',
 'verb_uwierzyć, że_',
 'verb_uwzględnić, że_',
 'verb_uzasadniać, że_',
 'verb_uzmysławiać sobie, że_',
 'verb_uznawać, że_',
 'verb_uznać, że_',
 'verb_uznać, że_2',
 'verb_uświadamiać komuś, że_',
 'verb_uświadamiać sobie, że_',
 'verb_uświadomić komuś, że_',
 'verb_uświadomić sobie, że_',
 'verb_wciskać, że_',
 'verb_wiadomo, że_',
 'verb_wiedzieć, że_',
 'verb_wierzyć, że_',
 'verb_wmawiać, że_',
 'verb_wmówić, że_',
 'verb_wnioskować, że_',
 'verb_wnosić z czegoś, że_',
 'verb_wskazywać, że_',
 'verb_wspominać, że2_',
 'verb_wspomnieć, że1_',
 'verb_wspomnieć, że2_',
 'verb_wybrzydzać, że_',
 'verb_wychodzi na to, że_',
 'verb_wychodzić komuś, że_',
 'verb_wychodzić z założenia, że_',
 'verb_wyczytać, że_',
 'verb_wydaje się komuś, że_',
 'verb_wydaje się, że_',
 'verb_wygląda na to, że_',
 'verb_wyjaśniać, że_',
 'verb_wyjaśnić, że_',
 'verb_wyjść na jaw, że_',
 'verb_wykazać, że_',
 'verb_wykazywać, że_',
 'verb_wykluczać, że_',
 'verb_wyliczyć, że_',
 'verb_wynikać, że_',
 'verb_wyobrazić sobie, że_1',
 'verb_wyobrazić sobie_2',
 'verb_wyobrażać sobie, że_1',
 'verb_wyobrażać sobie, że_2',
 'verb_wypisywać, że_',
 'verb_wypominać, że_',
 'verb_wystarczy, że_',
 'verb_wystarczyło, że_',
 'verb_wyszło na to, że_',
 'verb_wytykać, że_',
 'verb_wytłumaczyć, że_',
 'verb_wyznać, że_',
 'verb_wątpić, że_',
 'verb_zadeklarować, że_',
 'verb_zakładać, że_',
 'verb_zapamiętać, że_',
 'verb_zapewniać, że_',
 'verb_zapewnić, że_',
 'verb_zapisać, że_',
 'verb_zaplanować, że_',
 'verb_zapominać, że_',
 'verb_zapomnieć, że_',
 'verb_zapowiadać się, że_',
 'verb_zapowiadać, że_',
 'verb_zapowiedzieć, że_',
 'verb_zaproponować, że_',
 'verb_zaprzeczyć, że_',
 'verb_zarzekać się, że_',
 'verb_zarzucać, że_',
 'verb_zarzucić, że_',
 'verb_zaręczać, że_',
 'verb_zastrzec, że_',
 'verb_zastrzegać, że_',
 'verb_zasugerować, że_',
 'verb_zawiadamiać, że_',
 'verb_zaznaczać, że_',
 'verb_zaznaczyć, że_',
 'verb_założyć się, że_',
 'verb_założyć, że_',
 'verb_zaświadczać, że_',
 'verb_zdaje się [komuś], że_',
 'verb_zdarzać się, że_',
 'verb_zdarzyć się, że_',
 'verb_zdawać się, że_',
 'verb_zdawać sobie sprawę, że_',
 'verb_zdać sobie sprawę, że_',
 'verb_zdecydować się, że_',
 'verb_zdecydować, że_',
 'verb_zdziwić się, że_',
 'verb_zeznać, że_',
 'verb_zgadzać się, że_',
 'verb_zgodzić się, że_',
 'verb_zgłaszać, że_',
 'verb_zobowiązać się, że_',
 'verb_zorientować się, że_',
 'verb_zreflektować się, że_',
 'verb_zrozumieć, że_',
 'verb_zważyć, że_',
 'verb_zwracać komuś uwagę, że_',
 'verb_zwrócić na coś czyjąś uwagę, że_',
 'verb_zwąchać, że_',
 'verb_łudzić się, że_',
 'verb_śmiać się [z tego], że_',
 'verb_śmiać się z kogoś, że_',
 'verb_śmiać się, że_',
 'verb_śnić się, że_',
 'verb_świadczyć, że_',
 'verb_żalić się, że_',
 'verb_żartować, że_',
 'verb_żałować, że_',
 'verb_nan',
 'verb - main semantic class_epistemiczny',
 'verb - main semantic class_inne',
 'verb - main semantic class_mówienia',
 'verb - main semantic class_percepcyjny',
 'verb - main semantic class_nan',
 'verb - tense_past',
 'verb - tense_present',
 'verb - tense_nan',
 'verb - factive/nonfactive_NF',
 'verb - factive/nonfactive_nan',
 'complement - tense_inne',
 'complement - tense_past',
 'complement - tense_present',
 'complement - tense_nan',
 'T - type of sentence_indywidualne',
 'T - type of sentence_kontrfaktyczne',
 'T - type of sentence_performatyw',
 'T - type of sentence_pytajne',
 'T - type of sentence_reguła',
 'T - type of sentence_warunkowe',
 'T - type of sentence_nan']