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
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
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
import wandb

from collections import Counter

from utils import Instance, NLPDataset, Vocab
from utils import load_data_from_csv, save_frequencies, load_train_frequencies, generate_embedding_matrix, pad_collate_fn
from sklearn.metrics import accuracy_score, f1_score
"""================================ Task 01 ====================================================="""


# # # # # # # # # # # save_frequencies(train_instances) # save the training frequencies as dictionaries
# # # # # # # # frequencies = load_train_frequencies() # load the training frequencies as dictionaries

def process_data(text_vocab, batch_size, shuffle= True):

    NLP_instance = NLPDataset(text_vocab)
    # print(train_dataset[3])


    train_file_path = "data/sst_train_raw.csv"
    val_file_path   = "data/sst_valid_raw.csv"
    test_file_path  = "data/sst_test_raw.csv"

    train_dataset = NLP_instance.from_file(train_file_path)
    val_dataset = NLP_instance.from_file(val_file_path)
    test_dataset = NLP_instance.from_file(test_file_path)


    batch_size = batch_size # Only for demonstrative purposes
    shuffle = shuffle # Only for demonstrative purposes
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=shuffle, collate_fn=pad_collate_fn, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                                shuffle=shuffle, collate_fn=pad_collate_fn, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, 
                                shuffle=shuffle, collate_fn=pad_collate_fn, drop_last=True)
    # pdb.set_trace()
    # texts, labels, lengths_before_padding = next(iter(train_dataloader))
    # print(f"Texts: {texts}") # dim = (batch_size x max length of sentence in each batch)
    # print(f"Labels: {labels}")
    # print(f"Lengths: {lengths_before_padding}") # different length before padding. After padding all will be of same length.
    # pdb.set_trace()

    return train_dataloader, val_dataloader, test_dataloader



"""================================ Task 02 ====================================================="""


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_layer):
        super(GRUModel, self).__init__()

        self.embedding_layer = embedding_layer
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)

        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):

        lengths = torch.sum(x != 0, dim=1)
        # pdb.set_trace()
        embeds = self.embedding_layer(x)
        # mask = (x != 0).float()
        # masked_embeds = embeds * mask.unsqueeze(-1)
        # pdb.set_trace()
        
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)

        _, hidden = self.rnn(packed_embeds)
        # hidden is hidden state of the last layer of the RNN
        
        last_hidden = hidden[-1]  # Taking the hidden state of the last layer # [10, 150]
        # last_hidden = self.dropout(last_hidden)
        # pdb.set_trace()
        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train(model, train_loader, optimizer, criterion, gradient_clip, epoch, record):
    model.train() # enable dropout
    # texts, labels, lengths = next(iter(train_loader))
    # pdb.set_trace()
    total_loss = 0.0
    for batch_texts, batch_labels, batch_lengths in train_loader:
        # pdb.set_trace()
        optimizer.zero_grad()
        logits = model(batch_texts)
        # pdb.set_trace()
        loss = criterion(logits, batch_labels.float().view(10,1))
        total_loss += loss.item()
        loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        optimizer.step()
    wandb.log({"Train loss": total_loss / len(train_loader)}) if record else None
    # print(f'Epoch {epoch+1}: Train loss = {total_loss / len(train_loader)}')
    # print(f'Epoch {epoch}: Train loss = {loss.item()}')



def evaluate(model, dataset_loader, criterion, epoch=1, test=False, record=False):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for val_batch_texts, val_batch_labels, batch_lengths in dataset_loader:

            outputs = model(val_batch_texts)
            loss = criterion(outputs, val_batch_labels.float().view(10,1))
            total_loss += loss.item()
            # Compute metrics (accuracy, F1 score, confusion matrix) here if needed
            predicted = torch.round(torch.sigmoid(outputs)).squeeze().int()  # Sigmoid for binary predictions
            # pdb.set_trace()

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(val_batch_labels.cpu().numpy().tolist())
    
    accuracy = accuracy_score(all_labels, all_preds)
    if test:
        print(f'Test accuracy = {accuracy}')
    else:
        print(f'Epoch {epoch+1}: valid accuracy = {accuracy}')

    f1 = f1_score(all_labels, all_preds)
    confusion = confusion_matrix(all_labels, all_preds)
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(preds=all_preds,
                        y_true=all_labels,
                        class_names=["positive", "negative"])}) if record else None

    Loss= total_loss / len(dataset_loader)
    # print(f"Validation Loss: {total_loss / len(val_loader)}")
    # print(f"Accuracy: {accuracy}")
    # print(f"F1 Score: {f1}")
    # print(f"Confusion Matrix:\n{confusion}")
    wandb.log({"Accuracy": accuracy, "f1-score": f1, "Confusion Matrix": confusion, "Loss": Loss}) if record else None


def main(args):

    text_frequencies = load_train_frequencies()
    text_vocab = Vocab(text_frequencies)
    # pdb.set_trace()

    # # # embedding file path 
    embedding_file_path = 'data/sst_glove_6b_300d.txt'

    # (Vxd)
    embedding_layer, _, _ = generate_embedding_matrix(text_vocab, embedding_file_path)
    # print(embedding_layer) # (14806, 300) = (Vxd)

    # Define your hyperparameters
    input_size = 300
    hidden_size = 150
    learning_rate = args['lr']
    epochs = args['epochs']
    batch_size = args['batch_size']
    gradient_clip = args['clip']
    record = args['record']

    # Initialize model
    model = GRUModel(input_size, hidden_size, embedding_layer)

    train_dataloader, val_dataloader, test_dataloader = process_data(text_vocab, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.double()
        train(model, train_dataloader, optimizer, criterion, gradient_clip, epoch, record=record)
        evaluate(model, val_dataloader, criterion, epoch, record=record)

    print("Evaluation on Test Set:")
    evaluate(model, test_dataloader, criterion, test=True, record=record)
    wandb.finish()

if __name__ == '__main__':

    args = {}
    args['lr']=1e-4
    # args['seed']=7052020
    args['seed']=7568426
    args['batch_size']=10 
    args['epochs'] = 5
    args['clip']=0.25
    args['record']=False

    seed = args['seed']
    torch.manual_seed(seed)
    # np.random.seed(seed)
    
    if args['record']:
        wandb.init(
        # Set the project where this run will be logged
        project="Lab_03_Task_03", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"Experiment_05", 
        # Track hyperparameters and run metadata
        config={
                "learning_rate": args['lr'],
                "seed": args['seed'],
                "batch_size": args['batch_size'],
                "epochs": args['epochs'],
                "clip": args['clip'],
                })
    
    main(args)