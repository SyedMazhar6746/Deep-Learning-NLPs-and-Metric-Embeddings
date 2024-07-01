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

    return train_dataloader, val_dataloader, test_dataloader



"""================================ Task 02 ====================================================="""

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_layer, rnn_cell_type, use_attention, dropout, bidirectional):
        super(RNNModel, self).__init__()
        self.embedding_layer = embedding_layer
        self.use_attention = use_attention
        self.bidirectional = bidirectional  # Store bidirectional parameter
        
        num_directions = 2 if bidirectional else 1  # Adjust num_directions for hidden size
        
        if rnn_cell_type == "Vanilla":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            rnn_output_dim = hidden_size * num_directions
            
        elif rnn_cell_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            rnn_output_dim = hidden_size * num_directions
            
        elif rnn_cell_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            rnn_output_dim = hidden_size * num_directions
            
        else:
            raise ValueError("Invalid rnn_cell_type. Choose from 'Vanilla', 'GRU', or 'LSTM'.")
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(rnn_output_dim, hidden_size)  # Modify the output size of fc1 to 150
        self.fc2 = nn.Linear(hidden_size, 1)  # Modify the input size of fc2 to 150
        
        if use_attention:
            self.W1 = nn.Linear(rnn_output_dim, hidden_size // 2)   # size (300, 75)
            self.w2 = nn.Linear(hidden_size // 2, 1) # size (75, 1)
        
    def forward(self, x):
        lengths = torch.sum(x != 0, dim=1)
        embeds = self.embedding_layer(x)
        mask = (x != 0).float()
        masked_embeds = embeds * mask.unsqueeze(-1)
        
        packed_embeds = nn.utils.rnn.pack_padded_sequence(masked_embeds, lengths, batch_first=True, enforce_sorted=False)
        
        if isinstance(self.rnn, nn.LSTM):
            _, (hidden, _) = self.rnn(packed_embeds)
        else:
            _, hidden = self.rnn(packed_embeds)
        
        # Concatenate the forward and backward hidden states if bidirectional
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        if self.use_attention:
            # pdb.set_trace()
            attn_weights = self.w2(torch.tanh(self.W1(hidden)))  # (10, 1)
            attn_weights = torch.softmax(attn_weights, dim=0)    # (10, 1)
            attn_weights_un = attn_weights.unsqueeze(1).float()  # (10, 1, 1)
            hidden_un = hidden.unsqueeze(1).float()             # (10, 1, 300)
            attn_output = torch.bmm(attn_weights_un, hidden_un) # (10, 1, 300)
            x = attn_output.squeeze(1).double()                # (10, 300)
        else:
            x = hidden
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train(model, train_loader, optimizer, criterion, gradient_clip, epoch, record):
    model.train() # enable dropout
    total_loss = 0.0
    for batch_texts, batch_labels, batch_lengths in train_loader:
        optimizer.zero_grad()
        logits = model(batch_texts)
        loss = criterion(logits, batch_labels.float().view(10,1))
        total_loss += loss.item()
        loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        optimizer.step()
    wandb.log({"Train loss": total_loss/len(train_loader)}) if record else None
    print(f'Epoch {epoch}: Train loss = {total_loss/len(train_loader)}')

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
    # print(f"Validation Loss: {total_loss / len(dataset_loader)}")
    # print(f"Accuracy: {accuracy}")
    # print(f"F1 Score: {f1}")
    # print(f"Confusion Matrix:\n{confusion}")
    wandb.log({"Accuracy": accuracy, "f1-score": f1, "Confusion Matrix": confusion, "Loss": Loss}) if record else None

def main(args):

    text_frequencies = load_train_frequencies()
    text_vocab = Vocab(text_frequencies)

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

    # rnn cells
    vanilla = args["rnn_cell"][0]
    gru = args["rnn_cell"][1]
    lstm = args["rnn_cell"][2]

    # attention
    yes_at = args["attention"][0]
    no_at = args["attention"][1]

    # dropout
    yes_dr = args["dropout"][0]
    no_dr = args["dropout"][1]
    # pdb.set_trace()

    # bidirectional
    yes_bi = args["bidirectional"][0]
    no_bi = args["bidirectional"][1]

    # Initialize model
    model = RNNModel(input_size, hidden_size, embedding_layer, rnn_cell_type=lstm, use_attention=yes_at, dropout=yes_dr, bidirectional=yes_bi)

    train_dataloader, val_dataloader, test_dataloader = process_data(text_vocab, batch_size=batch_size, shuffle=True)



    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.double() # change the data to float64
        train(model, train_dataloader, optimizer, criterion, gradient_clip, epoch, record=record)
        evaluate(model, val_dataloader, criterion, epoch, record=record)

    print("Evaluation on Test Set:")
    evaluate(model, test_dataloader, criterion, test=True, record=record)
    wandb.finish()

if __name__ == '__main__':

    args = {}
    args['lr']               =1e-4
    args['seed']            =7052020
    args['batch_size']      =10 
    args['epochs']          = 5
    args['clip']            =0.25
    args['rnn_cell']        =["Vanilla", "GRU", "LSTM"]
    args['attention']       =[True, False]
    args['dropout']         = [0.2, 0.0]
    args['bidirectional']   = [True, False]
    args['record']=False

    seed = args['seed']
    torch.manual_seed(seed)
    # np.random.seed(seed)

    if args['record']:
        wandb.init(
        # Set the project where this run will be logged
        project="Lab_03_Bonus", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"Experiment_01", 
        # Track hyperparameters and run metadata
        config={
                "learning_rate": args['lr'],
                "seed": args['seed'],
                "batch_size": args['batch_size'],
                "epochs": args['epochs'],
                "clip": args['clip'],
                "rnn_cell": args['rnn_cell'],
                "attention": args['attention'],
                "dropout": args['dropout'],
                "bidirectional": args['bidirectional']
                })
    
    main(args)