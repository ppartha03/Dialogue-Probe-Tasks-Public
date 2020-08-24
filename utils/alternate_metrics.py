import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.meteor_score import meteor_score
import math
import argparse
from rouge import Rouge
import _pickle as cPickle
from rouge import Rouge
import logging
import csv
import math

bleu_met = nltk.translate.bleu_score.sentence_bleu
R = Rouge()
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='MultiWoZ')
args = parser.parse_args()

def Metrics(file_loc, increment=4, embedding_dict=None):
    rouge_p = []
    rouge_r = []
    rouge_f = []
    bleu = []
    fp = open(file_loc)
    D = fp.readlines()
    r_pre = 0.0
    r_rec = 0.0
    r_f1 = 0.0
    bert = 0.0
    sent_bleu = 0.0
    meteor_s = 0.0
    cnt_ = 0
    i=0
    while i<len(D):
        tar = D[i+2].split()[1:]
        mod = D[i+1].split()[1:]

        if '<eor>' in tar:
            ind_tar = tar.index('<eor>')
        else:
            ind_tar = -1
        if '<eor>' in mod:
            ind_mod = mod.index('<eor>')
        else:
            ind_mod = -1
        tar_embs = []
        mod_embs = []
        for word in tar[:ind_tar]:
            if word in embedding_dict:
                tar_embs+=[embedding_dict[word]]
        tar_embs = np.stack(tar_embs)
        for word in mod[:ind_mod]:
            if word in embedding_dict:
                mod_embs+=[embedding_dict[word]]
        mod_embs = np.stack(mod_embs)
        tar_emb = np.sum(tar_embs,axis=0)
        mod_emb = np.sum(mod_embs,axis=0)
        bert -= np.mean((tar_emb-mod_emb)**2)
        r_scores = R.get_scores(' '.join(mod[:ind_mod]),' '.join(tar[:ind_tar]))
        sent_bleu += bleu_met([mod[:ind_mod]],tar[:ind_tar],(0.5,0.5))
        meteor_s += meteor_score([' '.join(mod[:ind_mod])],' '.join(tar[:ind_tar]))
        r_pre += r_scores[0]['rouge-l']['p']
        r_rec += r_scores[0]['rouge-l']['r']
        r_f1 += r_scores[0]['rouge-l']['f']
        i+=increment
        cnt_+=1
    return {'METEOR':meteor_s/float(cnt_),'BLEU': sent_bleu/float(cnt_),'F1': r_f1/float(cnt_),'BERT':bert/float(cnt_)}

if __name__ == '__main__':
    bestmodels = {'bilstm_attn':{'100':{'METEOR':{'score':-math.inf,'epoch':0}, \
    'BLEU':{'score':-math.inf,'epoch':0},'BERT':{'score':-math.inf,'epoch':0},'F1':{'score':-math.inf,'epoch':0}},\
    '101':{'METEOR':{'score':-math.inf,'epoch':0},'BLEU':{'score':-math.inf,'epoch':0},\
    'BERT':{'score':-math.inf,'epoch':0},'F1':{'score':-math.inf,'epoch':0}},\
    '102':{'METEOR':{'score':-math.inf,'epoch':0},'BLEU':{'score':-math.inf,'epoch':0},\
    'BERT':{'score':-math.inf,'epoch':0},'F1':{'score':-math.inf,'epoch':0}}},\
    'hred':{'100':{'METEOR':{'score':-math.inf,'epoch':0},'BLEU':{'score':-math.inf,'epoch':0},\
    'BERT':{'score':-math.inf,'epoch':0},'F1':{'score':-math.inf,'epoch':0}},\
    '101':{'METEOR':{'score':-math.inf,'epoch':0},'BLEU':{'score':-math.inf,'epoch':0},\
    'BERT':{'score':-math.inf,'epoch':0},'F1':{'score':-math.inf,'epoch':0}},\
    '102':{'METEOR':{'score':-math.inf,'epoch':0},'BLEU':{'score':-math.inf,'epoch':0},\
    'BERT':{'score':-math.inf,'epoch':0},'F1':{'score':-math.inf,'epoch':0}}}, \
    'seq2seq_attn':{'100':{'METEOR':{'score':-math.inf,'epoch':0},\
    'BLEU':{'score':-math.inf,'epoch':0},'BERT':{'score':-math.inf,'epoch':0},\
    'F1':{'score':-math.inf,'epoch':0}},'101':{'METEOR':{'score':-math.inf,'epoch':0},\
    'BLEU':{'score':-math.inf,'epoch':0},'BERT':{'score':-math.inf,'epoch':0},\
    'F1':{'score':-math.inf,'epoch':0}},'102':{'METEOR':{'score':-math.inf,'epoch':0},\
    'BLEU':{'score':-math.inf,'epoch':0},'BERT':{'score':-math.inf,'epoch':0},\
    'F1':{'score':-math.inf,'epoch':0}}}, \
    'seq2seq':{'100':{'METEOR':{'score':-math.inf,'epoch':0},\
    'BLEU':{'score':-math.inf,'epoch':0},'BERT':{'score':-math.inf,'epoch':0},\
    'F1':{'score':-math.inf,'epoch':0}},'101':{'METEOR':{'score':-math.inf,'epoch':0},\
    'BLEU':{'score':-math.inf,'epoch':0},'BERT':{'score':-math.inf,'epoch':0},\
    'F1':{'score':-math.inf,'epoch':0}},'102':{'METEOR':{'score':-math.inf,'epoch':0},\
    'BLEU':{'score':-math.inf,'epoch':0},'BERT':{'score':-math.inf,'epoch':0},\
    'F1':{'score':-math.inf,'epoch':0}}},'transformer':{'100':{'METEOR':{'score':-math.inf,'epoch':0},\
    'BLEU':{'score':-math.inf,'epoch':0},'BERT':{'score':-math.inf,'epoch':0},\
    'F1':{'score':-math.inf,'epoch':0}},'101':{'METEOR':{'score':-math.inf,'epoch':0},\
    'BLEU':{'score':-math.inf,'epoch':0},'BERT':{'score':-math.inf,'epoch':0},\
    'F1':{'score':-math.inf,'epoch':0}},'102':{'METEOR':{'score':-math.inf,'epoch':0},\
    'BLEU':{'score':-math.inf,'epoch':0},'BERT':{'score':-math.inf,'epoch':0},\
    'F1':{'score':-math.inf,'epoch':0}}}}
    samples_folder = os.path.join('..','Results',args.dataset,'Samples')
    emb_dict = {}
    target = open(os.path.join("Best_Models_"+args.dataset+".csv"), "a")
    fields = ['Model','Seed','METEOR Epoch','BLEU Epoch','BERT Epoch','F1 Epoch','METEOR Score','BLEU Score','BERT Score','F1 Score']
    writer = csv.DictWriter(target, fieldnames=fields)
    writer.writerow(dict(zip(['Model','Seed','METEOR Epoch','BLEU Epoch','BERT Epoch','F1 Epoch','METEOR Score','BLEU Score','BERT Score','F1 Score'], ['Model','Seed','METEOR Epoch','BLEU Epoch','BERT Epoch','F1 Epoch','METEOR Score','BLEU Score','BERT Score','F1 Score'])))
    with open(args.dataset+'_BERT.txt', 'r', encoding="utf8") as f:
        for line in f:
            values = line.split('<del>')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            emb_dict.update({word:coefs})
    for k,seeds in bestmodels.items():
        print(k)
        for seed in seeds.keys():
            dict_ = {}
            for i in range(25):
                sample_file = os.path.join(samples_folder,k+'_seed_'+seed,'samples_valid_'+str(i)+'.txt')
                metrics = Metrics(sample_file,embedding_dict=emb_dict)
                for metric in metrics.keys():
                    if metrics[metric] > bestmodels[k][seed][metric]['score']:
                        bestmodels[k][seed][metric]['epoch'] = i
                        bestmodels[k][seed][metric]['score'] = metrics[metric]
            dict_['Model'] = k
            dict_['Seed'] = seed
            dict_['METEOR Epoch'] = bestmodels[k][seed]['METEOR']['epoch']
            dict_['BLEU Epoch'] = bestmodels[k][seed]['BLEU']['epoch']
            dict_['F1 Epoch'] = bestmodels[k][seed]['F1']['epoch']
            dict_['BERT Epoch'] = bestmodels[k][seed]['BERT']['epoch']
            dict_['METEOR Score'] = bestmodels[k][seed]['METEOR']['score']
            dict_['BLEU Score'] = bestmodels[k][seed]['BLEU']['score']
            dict_['F1 Score'] = bestmodels[k][seed]['F1']['score']
            dict_['BERT Score'] = bestmodels[k][seed]['BERT']['score']
            writer.writerow(dict_)
