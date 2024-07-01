#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from Task_1_mnist_data import MNISTMetricDataset
from utils_editing import evaluate, compute_representations


PRINT_LOSS_N = 100

EVAL_ON_TEST = True
EVAL_ON_TRAIN = True

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # Flatten the image tensor
        feats = img.view(img.size(0), -1)
        return feats

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "./data"
    ds_train = MNISTMetricDataset(mnist_download_root, split='train')
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader = DataLoader(ds_train, batch_size=64, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
    traineval_loader = DataLoader(ds_traineval, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

    # Initialize the IdentityModel
    model = IdentityModel().to(device)


    representations = compute_representations(model, train_loader, num_classes, device)
    if EVAL_ON_TRAIN:
        print("Evaluating on Validation set...")
        acc1 = evaluate(model, representations, traineval_loader, device)
        print(f"Eval Accuracy: {acc1 * 100:.2f}%")
    if EVAL_ON_TEST:
        print("Evaluating on test set...")
        acc1 = evaluate(model, representations, test_loader, device)
        print(f"Test Accuracy: {acc1 * 100:.2f}%")


