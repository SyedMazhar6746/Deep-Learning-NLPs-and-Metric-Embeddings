#!/usr/bin/python3

import torch
from torch import nn
from pathlib import Path
import os
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


class BaselineModel(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_layer, embed_shape_0, embed_shape_1):
        super(BaselineModel, self).__init__()

        self.embedding_layer = embedding_layer
        # self.embedding_layer = nn.Embedding(embed_shape_0, embed_shape_1)

        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, x):
        x = x
        embeds = self.embedding_layer(x) #(10, 34, 300) # 34 will change as per sentence length 
        x = embeds.sum(axis=1) / (embeds != 0).sum(axis=1).clamp(min=1).float()
        # kernel_size = embeds.size(1)
        # avg_pool_layer = nn.AvgPool1d(kernel_size)
        # x = avg_pool_layer(embeds.permute(0, 2, 1)).squeeze()

        # pdb.set_trace()
        # Masking padded values
        # mask = (x != 0).float()  # Create a mask where non-zero values are 1 and padded values are 0
        # masked_embeds = embeds * mask.unsqueeze(-1)  # Apply the mask to the embeddings and getting rid of zeros in multiplication

        # x = self.avg_pool(masked_embeds.permute(0, 2, 1)).squeeze()  # (10, 300)
        # x = self.avg_pool(embeds.permute(0, 2, 1)).squeeze()  # (10, 300)
        x = self.fc1(x) # expects input dim = 300

        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def train(model, train_loader, optimizer, criterion, gradient_clip, epoch, record):
    model.train() # enable dropout
    # texts, labels, lengths = next(iter(train_loader))
    # pdb.set_trace()
    total_loss = 0.0
    for texts, labels, lengths in train_loader: # batch wise training
        # pdb.set_trace()
        optimizer.zero_grad()
        logits = model(texts)
        # pdb.set_trace()
        # loss = criterion(logits, labels.float().unsqueeze(1))
        loss = criterion(logits, labels.float().view(10,1))
        total_loss += loss.item()
        loss.backward()
        # clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    wandb.log({"Train loss": total_loss / len(train_loader)}) if record else None
    # print the total loss
    # print(f'Epoch {epoch}: Train loss = {total_loss / len(train_loader)}')


def evaluate(model, dataset_loader, criterion, epoch=1, mode='test', record=False):
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
    print(f'Epoch {epoch}: {mode} accuracy = {accuracy}')

    confusion = confusion_matrix(all_labels, all_preds)

    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(preds=all_preds,
                            y_true=all_labels,
                            class_names=["positive", "negative"])}) if record else None

    f1 = f1_score(all_labels, all_preds)

    Loss= total_loss / len(dataset_loader)
    # print(f"{mode} Loss: {total_loss / len(dataset_loader)}")
    # print(f"{mode} F1 Score: {f1}")
    # print(f"{mode} Confusion Matrix:\n{confusion}")
    # print(f"Accuracy: {accuracy}")

    # Log metrics from your script to W&B
    wandb.log({"Accuracy": accuracy, "f1-score": f1, "Confusion Matrix": confusion, "Loss": Loss}) if record else None

def main(args):

    text_frequencies = load_train_frequencies()
    text_vocab = Vocab(text_frequencies)
    # pdb.set_trace()

    # # # embedding file path 
    embedding_file_path = 'data/sst_glove_6b_300d.txt'

    # (Vxd)
    embedding_layer, embed_shape_0, embed_shape_1 = generate_embedding_matrix(text_vocab, embedding_file_path)
    # print(embedding_layer) # (14806, 300) = (Vxd)

    # Define your hyperparameters
    input_size          = 300
    hidden_size         = 150
    learning_rate       = args['lr']
    epochs              = args['epochs']
    batch_size          = args['batch_size']
    gradient_clip       = args['clip']
    record              = args['record']


    # Initialize model
    model = BaselineModel(input_size, hidden_size, embedding_layer, embed_shape_0, embed_shape_1)

    train_dataloader, val_dataloader, test_dataloader = process_data(text_vocab, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.double()
        train(model, train_dataloader, optimizer, criterion, gradient_clip, epoch, record)
        evaluate(model, val_dataloader, criterion, epoch, mode='val', record=record)

    print("Evaluation on Test Set:")
    evaluate(model, test_dataloader, criterion)
    wandb.finish()

if __name__ == '__main__':

    args = {}
    args['lr']=1e-4
    # args['seed']=7052020
    args['seed']=7654219
    args['batch_size']=10 
    args['epochs'] = 5
    args['clip']=0.25
    args['record']=False

    if args['record']:
        wandb.init(
        # Set the project where this run will be logged
        project="Lab_03_Task_02", 
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
    seed = args['seed']
    torch.manual_seed(seed)
    # np.random.seed(seed)
    main(args)
