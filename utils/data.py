from abc import ABC
from utils.io import read_pkl, read_file

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import pickle
import os
import numpy as np


class Data(Dataset):
    def __init__(self, dialog, context_len=128, response_len=128, decoder_start_token_id=1):
        super(Dataset, self).__init__()
        self.context = [sum(item[:-1], []) for item in dialog]
        self.response = [item[-1] for item in dialog]

        self.context_len = context_len
        self.response_len = response_len

        self.decoder_start_token_id = decoder_start_token_id

    def __getitem__(self, index):
        context = torch.tensor(self.context[index][-self.context_len:])
        response = torch.tensor([self.decoder_start_token_id] + self.response[index][:self.response_len])

        return context, response

    def __len__(self):
        return len(self.context)


def collate_fn(data):
    pad_idx = 0
    decoder_start_token_id = 1
    context, response = zip(*data)
    return {
        'context': pad_sequence(context, batch_first=True, padding_value=pad_idx),
        'response': pad_sequence(response, batch_first=True, padding_value=pad_idx),
        'pad_idx': pad_idx,
        'decoder_start_token_id': decoder_start_token_id
    }
