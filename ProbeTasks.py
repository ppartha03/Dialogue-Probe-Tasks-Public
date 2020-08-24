from torchtext import data
from modelbase.rnn import RecurrentEncoder, Encoder, AttnDecoder, Decoder
from modelbase.transformer import TransformerModel
from dataset_utils.data_iterator import MultiWoZ, PersonaChat
import torch.optim as optim
from utils.eval_metric import getBLEU
from utils.optim import GradualWarmupScheduler
from utils.transformer_utils import create_masks
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import pandas as pd
from seq2seq import Seq2Seq
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import logging
import wandb
import random
import math
import csv
import sys
import matplotlib.pyplot as plt
import itertools

# commandline arguments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--s2s_hidden_size', type=int, default = 256)
parser.add_argument('--s2s_embedding_size',type=int, default = 128)
parser.add_argument('--transformer_dropout',type=float, default = 0.2)
parser.add_argument('--transformer_hidden_dim',type=int, default = 200)
parser.add_argument('--transformer_embedding_dim',type=int, default = 200)
parser.add_argument('--transformer_n_layers',type=int, default = 2)
parser.add_argument('--transformer_n_head',type=int, default = 2)
parser.add_argument('--batch_size',type=int,default=8)

args = parser.parse_args()

def plotHexbin(LM,epoch,filename, X_r):
    fig, axs = plt.subplots(ncols=1, sharey=True, figsize=(7, 4))
    fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)

    hb = axs.hexbin(X_r[:,0], X_r[:,1], gridsize=80, bins='log', cmap='YlOrBr')
    #axs.axis([-8,10.5,-7,10])
    axs.set_title(LM +' context embeddings distribution after '+epoch+' epochs')
    cb = fig.colorbar(hb, ax=axs)
    cb.set_label('counts')
    plt.savefig(filename)

def probeTask(x_train, y_train, x_valid, y_valid):
    mlb = MultiLabelBinarizer().fit(y_train)
    y_train = mlb.transform(y_train)
    y_valid = mlb.transform(y_valid)
    clf = OneVsRestClassifier(LogisticRegression(random_state = 42))
    clf.fit(x_train, y_train[:8,:])
    y_pred = clf.predict(x_valid)
    return f1_score(y_pred, y_valid[:8,:], average = 'micro')

def getData(model,iterator,cid_dict):
    X = None
    context_ids = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.Context.to(device)
            trg = batch.Target.to(device)
            context_id = batch.context_id.view(-1)
            context_ids += [cid_dict[x] for x in context_id]
            if model.type == 'transformer':
                src = src.transpose(0,1)
                trg = trg.transpose(0,1)
                trg_input = trg[:,:-1]
                src_mask, trg_mask = create_masks(src, trg_input, pad_idx)
                hidden = model(src, src_mask, trg_input, trg_mask, probetask = True)
                if type(X) != type(None):
                    X = torch.cat([X,hidden])
                else:
                    X = hidden
            else:
                hidden = model(src,trg,0, probetask = True)  # turn off the teacher forcing
                if type(X) != type(None):
                    X = torch.cat([X,hidden])
                else:
                    X = hidden
            break
        return X, context_ids


def ProbeTasks(model, model_folder, epochs, modeltype, train_iterator, \
 valid_iterator, dataset, csv_train, csv_valid, cid_dict,inp_seed, run_id):
    ''' Evaluation loop for the model to evaluate.
    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        criterion: loss criterion.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    pca = PCA(n_components=2)
    flag = 0
    ANALYSIS_PATH = os.path.join('Results', dataset, 'Analsysis', run_id)
    CSV_PATH = os.path.join('Results', dataset, 'Analsysis')
    if not os.path.exists(ANALYSIS_PATH):
        os.makedirs(os.path.join(ANALYSIS_PATH,'Graphs'))
        if not os.path.exists(os.path.join(CSV_PATH,"Probe_Tasks_"+dataset+".csv")):
            flag = 1

    if dataset == 'MultiWoZ':
        all_tasks = ['contextID','AllTopics','filename', 'UtteranceIndex', 'Context', 'Target', 'ResponseLength',\
         'UtteranceLoc','RepeatInfo','RecentTopic','RecentSlots', 'RecentValues', \
         'NumRecentInfo','AllValues', 'AllSlots', 'NumAllInfo','NumRepeatSlots', 'NumAllTopics', 'IsMultiTask', \
         'EntitySlots', 'EntityValues', 'ActionSelect']
        all_tasks_ind = [1,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        fieldnames=['Model','Epoch','Seed']+[all_tasks[i] for i in all_tasks_ind]
    else:
        all_tasks = ['ContextID','filename','UtteranceIndex', 'Context', 'Target', 'ResponseLength',\
        'UtteranceLoc','PersonalInfo', 'WordCont']
        all_tasks_ind = [5,6,7,8]
        fieldnames=['Model','Epoch','Seed']+[all_tasks[i] for i in all_tasks_ind]
    target = open(os.path.join(CSV_PATH,"Probe_Tasks_"+dataset+".csv"), "a")
    writer = csv.DictWriter(target, fieldnames=fieldnames)
    if flag:
        writer.writerow(dict(zip(fieldnames, fieldnames)))
    assert type(epochs) == type(list())
    for epoch in epochs:

        probe_f1s = {'Model': modeltype, 'Epoch': epoch, 'Seed': inp_seed}
        model_file = os.path.join(model_folder,modeltype+'_'+str(epoch)+'.pt')
        model.load_state_dict(torch.load(model_file))
        model.eval()
        # loss
        # we don't need to update the model parameters. only forward pass.
        X_train, cid_train = getData(model, train_iterator, cid_dict)
        X_valid, cid_valid = getData(model, valid_iterator, cid_dict)
        X_train = torch.cat([x[1].view(1,-1) for x in sorted(zip(cid_train,X_train))], dim =0)#torch.cat(list(dict(sorted(zip(cid_train,X_train))).values()))
        X_valid = torch.cat([x[1].view(1,-1) for x in sorted(zip(cid_valid,X_valid))], dim =0)
        if inp_seed == 100:
            X_r = pca.fit_transform(X_train.cpu())
            plotHexbin(modeltype,str(epoch),os.path.join(ANALYSIS_PATH,'Graphs',modeltype+'_'+str(epoch+1)+'.png'),X_r)
        for task in all_tasks_ind:
            y_train = csv_train.get(task)
            y_valid = csv_valid.get(task)
            f1 = probeTask(X_train, y_train, X_valid, y_valid)
            probe_f1s.update({all_tasks[task]: f1})
        writer.writerow(probe_f1s)

if __name__ == '__main__':
    models = ['seq2seq','seq2seq_attn','bilstm_attn','hred','transformer']
    dataset = ['PersonaChat', 'MultiWoZ']
    seeds = [100,101,102] # seed values
    MAX_LENGTH = 101
    BATCH_SIZE = args.batch_size
    for inp_seed,inp_model,inp_dataset in itertools.product(seeds,models,dataset):
        print(inp_model, inp_dataset, inp_seed)
        np.random.seed(inp_seed)
        run_id = inp_model + "_seed_" + str(inp_seed)
        if inp_dataset == 'MultiWoZ':
            train_iterator, valid_iterator, test_iterator, pad_idx, INPUT_DIM, itos_vocab, itos_context_id = MultiWoZ(batch_size = BATCH_SIZE ,max_length = MAX_LENGTH)
            csv_train = pd.read_csv('./Dataset/MultiWoZ/MultiWoZ_train.csv', header = None,\
             converters = {i: lambda x: x.split() for i in range(1,23)}).sort_values(by=0)
            csv_valid = pd.read_csv('./Dataset/MultiWoZ/MultiWoZ_valid.csv', header = None,\
             converters = {i: lambda x: x.split() for i in range(1,23)}).sort_values(by=0)
        elif inp_dataset == 'PersonaChat':
            train_iterator, valid_iterator, test_iterator, pad_idx, INPUT_DIM, itos_vocab, itos_context_id = PersonaChat(max_length = MAX_LENGTH, batch_size = BATCH_SIZE)
            csv_train = pd.read_csv('./Dataset/PersonaChat/PersonaChat_train.csv', header = None,\
             converters = {i: lambda x: x.split() for i in range(1,9)}).sort_values(by=0)
            csv_valid = pd.read_csv('./Dataset/PersonaChat/PersonaChat_valid.csv', header = None,\
             converters = {i: lambda x: x.split() for i in range(1,9)}).sort_values(by=0)
        CLIP = 10               # gradient clip value    # directory name to save the models.
        MODEL_SAVE_PATH = os.path.join('Results', inp_dataset,'Model',run_id)

        OUTPUT_DIM = INPUT_DIM
        ENC_EMB_DIM = args.s2s_embedding_size # encoder embedding size
        DEC_EMB_DIM = args.s2s_embedding_size   # decoder embedding size (can be different from encoder embedding size)
        HID_DIM = args.s2s_hidden_size       # hidden dimension (must be same for encoder & decoder)
        N_LAYERS = 2        # number of rnn layers (must be same for encoder & decoder)
        HRED_N_LAYERS = 2
        ENC_DROPOUT = 0   # encoder dropout
        DEC_DROPOUT = 0   # decoder dropout (can be different from encoder droput)

        #TransformerParameters

        emsize = args.transformer_embedding_dim # 200 embedding dimension
        nhid = args.transformer_hidden_dim #200 the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = args.transformer_n_layers #2 the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = args.transformer_n_head #2 the number of heads in the multiheadattention models
        dropout = args.transformer_dropout # 0.2 the dropout value

        if inp_model == 'seq2seq':
            enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
            dec = Decoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT).to(device)
            model = Seq2Seq(enc, dec, attn = False).to(device)
            optimizer = optim.Adam(model.parameters())
        elif inp_model == 'hred':
            enc = RecurrentEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
            dec = AttnDecoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT,MAX_LENGTH).to(device)
            model = Seq2Seq(enc, dec, attn = True).to(device)
            optimizer = optim.Adam(model.parameters())
        elif inp_model == 'seq2seq_attn':
            enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HRED_N_LAYERS, ENC_DROPOUT).to(device)
            dec = AttnDecoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, HRED_N_LAYERS, DEC_DROPOUT,MAX_LENGTH).to(device)
            model = Seq2Seq(enc, dec, attn = True).to(device)
            optimizer = optim.Adam(model.parameters())
        elif inp_model == 'bilstm_attn':
            enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, bi_directional = True).to(device)
            dec = AttnDecoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT,MAX_LENGTH).to(device)
            model = Seq2Seq(enc, dec, attn = True).to(device)
            optimizer = optim.Adam(model.parameters())
        elif inp_model == 'transformer':
            model = TransformerModel(INPUT_DIM, emsize, nhead, nlayers, dropout).to(device).to(device)
            optimizer = optim.Adam(model.parameters(), lr = 0.01)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=2)
        else:
            print('Error: Model Not There')
            sys.exit(0)
        #epochs = range(0,25,2)
        ProbeTasks(model,MODEL_SAVE_PATH, [0,24,'best_bleu'], inp_model, train_iterator,\
         test_iterator, inp_dataset, csv_train, csv_valid, itos_context_id,inp_seed,run_id)
