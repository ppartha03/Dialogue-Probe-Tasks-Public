# credits to the machine translation code tutorial at https://ravirajag.dev/
from torchtext import data
from base_models.rnn import RecurrentEncoder, Encoder, AttnDecoder, Decoder
from base_models.transformer import TransformerModel
from dataset_utils.data_iterator import MultiWoZ
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import logging
import wandb
import random
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Seq2Seq(nn.Module):
    ''' This class contains the implementation of complete sequence to sequence network.
    It uses to encoder to produce the context vectors.
    It uses the decoder to produce the predicted target sentence.
    Args:
        encoder: A Encoder class instance.
        decoder: A Decoder class instance.
    '''
    def __init__(self, encoder, decoder, attn = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_attn = attn

    def forward(self, src, trg, teacher_forcing_ratio=1.0, probetask = False):
        # src is of shape [sequence_len, batch_size]
        # trg is of shape [sequence_len, batch_size]

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # to store the outputs of the decoder
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)

        # context vector, last hidden and cell state of encoder to initialize the decoder
        encoder_outputs, hidden, cell = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        hidden_enc = hidden
        if probetask:
            hidden_enc = hidden_enc.sum(0)
            return hidden_enc
        input = trg[0, :]
        if not self.encoder_attn:
            for t in range(1, max_len):
                output, hidden, cell = self.decoder(input, hidden, cell)
                outputs[t] = output
                use_teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.max(2)[1].squeeze(0)
                input = (trg[t] if use_teacher_force else top1)
        else:
            for t in range(1, max_len):
                output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
                outputs[t] = output
                use_teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.max(2)[1].squeeze(0)
                input = (trg[t] if use_teacher_force else top1)

        # outputs is of shape [sequence_len, batch_size, output_dim]
        hidden_enc = hidden_enc.sum(0).unsqueeze(0)
        return outputs, hidden_enc
