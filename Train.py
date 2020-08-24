from torchtext import data
from modelbase.rnn import RecurrentEncoder, Encoder, AttnDecoder, Decoder
from modelbase.transformer import TransformerModel
from dataset_utils.data_iterator import MultiWoZ, PersonaChat
import torch.optim as optim
from utils.eval_metric import getBLEU
from utils.optim import GradualWarmupScheduler
from utils.transformer_utils import create_masks
from Seq2Seq import Seq2Seq
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import logging
#import wandb
import random
import math
import csv
import sys

# commandline arguments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, default = '.')
parser.add_argument('--s2s_hidden_size', type=int, default = 256)
parser.add_argument('--s2s_embedding_size',type=int, default = 128)
parser.add_argument('--transformer_dropout',type=float, default = 0.2)
parser.add_argument('--transformer_hidden_dim',type=int, default = 512)
parser.add_argument('--transformer_embedding_dim',type=int, default = 512)
parser.add_argument('--transformer_n_layers',type=int, default = 2)
parser.add_argument('--transformer_n_head',type=int, default = 2)
parser.add_argument('--model',type=str,default='seq2seq')
parser.add_argument('--dataset',type=str,default='PersonaChat')
parser.add_argument('--batch_size',type=int,default=2)
parser.add_argument('--seed', type=int, default=100)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
def train(model, iterator, optimizer, criterion, clip, itos_vocab, itos_context_id):
    ''' Training loop for the model to train.
    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        optimizer: Optimizer for the model.
        criterion: loss criterion.
        clip: gradient clip value.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.train()
    # loss
    epoch_loss = 0
    id_to_hidden = {}
    for i, batch in enumerate(iterator):
        src = batch.Context.to(device)
        trg = batch.Target.to(device)

        optimizer.zero_grad()

        # trg is of shape [sequence_len, batch_size]
        # output is of shape [sequence_len, batch_size, output_dim]
        if model.type == 'transformer':
            src = src.transpose(0,1)
            trg = trg.transpose(0,1)
            trg_input = trg[:,:-1]
            src_mask, trg_mask = create_masks(src, trg_input, pad_idx)
            output,hidden = model(src, src_mask, trg_input, trg_mask)
            ys = trg[:,1:].contiguous().view(-1)
            loss = criterion(output.view(-1, output.size(-1)), ys)
        else:
            output,hidden = model(src, trg)
            loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))
        # loss function works only 2d logits, 1d targets
        # so flatten the trg, output tensors. Ignore the <sos> token
        # trg shape shape should be [(sequence_len - 1) * batch_size]
        # output shape should be [(sequence_len - 1) * batch_size, output_dim]
        # for b_id,hidden_state in zip(batch.context_id.squeeze(0),hidden.squeeze(0)):

        # backward pass
        loss.backward()

        # clip the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the parameters
        optimizer.step()

        epoch_loss += loss.item()
    # return the average loss
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, itos_vocab, itos_context_id, sample_saver):
    ''' Evaluation loop for the model to evaluate.
    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        criterion: loss criterion.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    model.eval()
    # loss
    epoch_loss = 0
    id_to_hidden = {}
    # we don't need to update the model parameters. only forward pass.
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.Context.to(device)
            trg = batch.Target.to(device)
            if model.type == 'transformer':
                src = src.transpose(0,1)
                trg = trg.transpose(0,1)
                trg_input = trg[:,:-1]
                src_mask, trg_mask = create_masks(src, trg_input, pad_idx)
                output,hidden = model(src, src_mask, trg_input, trg_mask)
                ys = trg[:,1:].contiguous().view(-1)
                loss = criterion(output.view(-1, output.size(-1)), ys)
                output = output.permute(1,0,2)
            else:
                output , hidden = model(src,trg,0)  # turn off the teacher forcing
                loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))
            top1 = output.max(2)[1].squeeze(0)
            for b_index in range(len(batch)):
                c = ' '.join([itos_vocab[idx.item()] for idx in batch.Context[:,b_index]])
                t = ' '.join([itos_vocab[idx.item()] for idx in batch.Target[:,b_index]])
                model_res = ' '.join([itos_vocab[idx.item()] for idx in top1[:,b_index]])
                sample_saver.write('Context: '+c +'\n'+'Model_Response: '+ model_res +'\n' +'Target: ' + t +'\n\n')
            # loss function works only 2d logits, 1d targets
            # so flatten the trg, output tensors. Ignore the <sos> token
            # trg shape shape should be [(sequence_len - 1) * batch_size]
            # output shape should be [(sequence_len - 1) * batch_size, output_dim]


            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


MAX_LENGTH = 101
BATCH_SIZE = args.batch_size
run_id = args.model + "_seed_" + str(args.seed)
if args.dataset == 'MultiWoZ':
    train_iterator, valid_iterator, test_iterator, pad_idx, INPUT_DIM, itos_vocab, \
     itos_context_id = MultiWoZ(batch_size = BATCH_SIZE ,max_length = MAX_LENGTH)
elif args.dataset == 'PersonaChat':
    train_iterator, valid_iterator, test_iterator, pad_idx, INPUT_DIM, itos_vocab, \
    itos_context_id = PersonaChat(batch_size = BATCH_SIZE, max_length = MAX_LENGTH)
N_EPOCHS = 25           # number of epochs
CLIP = 10               # gradient clip value    # directory name to save the models.
HIDDEN_SAVE_PATH = os.path.join('Results', args.dataset,'Hidden',run_id)
if not os.path.exists(HIDDEN_SAVE_PATH):
    os.makedirs(os.path.join(HIDDEN_SAVE_PATH,'Train'))
    os.makedirs(os.path.join(HIDDEN_SAVE_PATH,'Valid'))
MODEL_SAVE_PATH = os.path.join('Results', args.dataset,'Model',run_id)
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
SAMPLES_PATH = os.path.join('Results',args.dataset,'Samples',run_id)
if not os.path.exists(SAMPLES_PATH):
    os.makedirs(SAMPLES_PATH)
LOG_FILE_NAME = 'logs.txt'
logging.basicConfig(filename=os.path.join(MODEL_SAVE_PATH,LOG_FILE_NAME),
                        filemode='a+',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
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

# encoder
if args.model == 'seq2seq':
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
    dec = Decoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT).to(device)
    model = Seq2Seq(enc, dec, attn = False).to(device)
    optimizer = optim.Adam(model.parameters())
elif args.model == 'hred':
    enc = RecurrentEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
    dec = AttnDecoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT,MAX_LENGTH).to(device)
    model = Seq2Seq(enc, dec, attn = True).to(device)
    optimizer = optim.Adam(model.parameters())
elif args.model == 'seq2seq_attn':
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HRED_N_LAYERS, ENC_DROPOUT).to(device)
    dec = AttnDecoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, HRED_N_LAYERS, DEC_DROPOUT,MAX_LENGTH).to(device)
    model = Seq2Seq(enc, dec, attn = True).to(device)
    optimizer = optim.Adam(model.parameters())
elif args.model == 'bilstm_attn':
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, bi_directional = True).to(device)
    dec = AttnDecoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT,MAX_LENGTH).to(device)
    model = Seq2Seq(enc, dec, attn = True).to(device)
    optimizer = optim.Adam(model.parameters())
elif args.model == 'transformer':
    model = TransformerModel(INPUT_DIM, emsize, nhead, nlayers, dropout).to(device).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=2)
else:
    print('Error: Model Not There')
    sys.exit(0)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
sys.exit(0)

# loss function calculates the average loss per token
# passing the <pad> token to ignore_idx argument, we will ignore loss whenever the target token is <pad>
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)

best_validation_bleu = float('-inf')

for epoch in range(N_EPOCHS):
    sample_saver_eval = open(os.path.join(SAMPLES_PATH,"samples_valid_" +str(epoch) +'.txt'),'w')
    hidden_saver_train = open(os.path.join(HIDDEN_SAVE_PATH,'Train',str(epoch)+'.txt'),'w')
    hidden_saver_eval = open(os.path.join(HIDDEN_SAVE_PATH,'Valid',str(epoch)+'.txt'),'w')
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP, itos_vocab, itos_context_id )
    valid_loss = evaluate(model, valid_iterator, criterion, itos_vocab, itos_context_id, sample_saver_eval)
    sample_saver_eval.close()
    valid_bleu = getBLEU(sample_saver_eval.name)
    logging.info(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | \
    Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | \
    Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU: {valid_bleu:7.3f} |')
    print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | \
    Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | \
    Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU: {valid_bleu:7.3f} |')
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH,args.model+'_'+str(epoch)+'.pt'))
    if valid_bleu > best_validation_bleu:
        best_validation_bleu = valid_bleu
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH,args.model+'_best_bleu.pt'))

# Test set performance

model_file = os.path.join(MODEL_SAVE_PATH,modeltype+'_best_bleu.pt')
model.load_state_dict(torch.load(model_file))
sample_saver_test = open(os.path.join(SAMPLES_PATH,'samples_test.txt'),'w')
test_loss = evaluate(model, test_iterator, criterion, itos_vocab, itos_context_id, sample_saver_test)
test_bleu = getBLEU(sample_saver_eval.name)
print(f'| Epoch: {epoch+1:03} | Test. Loss: {test_loss:.3f} | Test. PPL: {math.exp(test_loss):7.3f} | Test. BLEU: {test_bleu:7.3f} |')
logging.info(f'| Epoch: {epoch+1:03} | Test. Loss: {test_loss:.3f} | Test. PPL: {math.exp(test_loss):7.3f} | Test. BLEU: {test_bleu:7.3f} |')
