#!/usr/bin/python3

import torch
from torch import nn
from pathlib import Path
import os
# from torch.utils.tensorboard import SummaryWriter
import skimage as ski
import math
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Subset, DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import pdb
import skimage as ski
import skimage.io


from dataclasses import dataclass
from typing import List

import csv


from collections import Counter

from utils import Instance, NLPDataset, Vocab
from utils import load_data_from_csv, save_frequencies, load_frequencies, generate_embedding_matrix#, pad_collate_fn

file_path = "data/sst_train_raw.csv"

frequencies = load_frequencies()
text_vocab = Vocab(frequencies)

train_set = NLPDataset(text_vocab, file_path)
new_set = train_set.from_file()
print(new_set[3])

# # # # # # # # file_path = "data/"
# # # # # # # # # Load your train, test, and val datasets
# # # # # # # # train_instances, train_text_freq, train_label_freq = load_data_from_csv(file_path+'sst_train_raw.csv')
# # # # # # # # # val_instances, val_text_freq, val_label_freq = load_data_from_csv(file_path+'sst_valid_raw.csv')
# # # # # # # # # test_instances, test_text_freq, test_label_freq = load_data_from_csv(file_path+'sst_test_raw.csv')
# # # # # # # # # pdb.set_trace()
# # # # # # # # # # # """Extract text and label"""
# # # # # # # # i=1
# # # # # # # # for instance in train_instances:
# # # # # # # #     # pdb.set_trace()
# # # # # # # #     text, label = instance.text, instance.label
# # # # # # # #     print(f"text: {text}")
# # # # # # # #     print(f"Label: {label}") # _positive = 8 length
# # # # # # # #     i+=1
# # # # # # # #     if i ==4 :
# # # # # # # #         break



# # # # # # # # # # # save_frequencies(train_instances) # save the training frequencies as dictionaries
# # # # # # # # frequencies = load_frequencies() # load the training frequencies as dictionaries

# # # # # # # # # pdb.set_trace()
# # # # # # # # # # # """How to extract the count of texts and labels"""
# # # # # # # # text_vocab = Vocab(frequencies)
# # # # # # # # numericalized_text = text_vocab.encode(text)
# # # # # # # # numericalized_label = text_vocab.encode(label)
# # # # # # # # print('numericalized_text:', numericalized_text)
# # # # # # # # print('numericalized_label:', numericalized_label)


# # # embedding file path 
# embedding_file_path = 'data/sst_glove_6b_300d.txt'


# embedding_layer = generate_embedding_matrix(text_vocab, embedding_file_path)
# print(embedding_layer)












"""============================================================================="""
# def pad_collate_fn(batch): # new function
#     texts, labels = zip(*batch)
#     lengths = torch.tensor([len(text) for text in texts])
#     texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
#     return texts, torch.tensor(labels), lengths



# # # all_texts = [instance.text for instance in train_instances]
# # # all_labels = [instance.label for instance in train_instances]


# # # def pad_collate_fn(batch):
# # #     texts, labels = zip(*batch)
# # #     lengths = torch.tensor([len(text) for text in texts])
# # #     texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
# # #     return texts, torch.tensor(labels), lengths


# # # batch_size = 2 # Only for demonstrative purposes
# # # shuffle = False # Only for demonstrative purposes

# # train_dataset = NLPDataset.from_file('data/sst_train_raw.csv')

# # # train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
# # #                               shuffle=shuffle, collate_fn=pad_collate_fn)
# # # # pdb.set_trace()
# # # texts, labels, lengths = next(iter(train_dataloader))
# # # print(f"Texts: {texts}")
# # # print(f"Labels: {labels}")
# # # print(f"Lengths: {lengths}")


