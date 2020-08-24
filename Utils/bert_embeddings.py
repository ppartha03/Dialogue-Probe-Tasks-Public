import sys
#sys.path.append('../DatasetUtils')
from DatasetUtils.DataIterator import MultiWoZ, PersonaChat
from bert_embedding import BertEmbedding
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'PersonaChat')
args = parser.parse_args()

if args.dataset == 'MultiWoZ':
    _, _, _, _, _, vocab, _ = MultiWoZ(batch_size = 32 ,max_length = 100)
elif args.dataset == 'PersonaChat':
    _, _, _, _, _, vocab, _ = PersonaChat(batch_size = 32, max_length = 100)

f = open(args.dataset+'_BERT.txt','w')
m = 0
bert = BertEmbedding()
for i in range(len(vocab)):
    print(i,'/',len(vocab))
    st = vocab[i]+'<del>'
    if vocab[i] in ['<eor>','<eou>','<pad>','<sos>']:
        st += '<del>'.join([str(0.0)]*768)
    else:
        emb = bert([vocab[i]])
        st+= '<del>'.join([str(w) for w in emb[0][1][0].tolist()])
    f.write(st+'\n')
f.close()
