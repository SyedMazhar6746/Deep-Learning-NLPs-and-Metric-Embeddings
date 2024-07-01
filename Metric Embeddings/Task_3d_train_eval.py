#!/usr/bin/python3

import os
import time
import torch.optim
from Task_1_mnist_data import MNISTMetricDataset
from torch.utils.data import DataLoader
from Task_2_metric import SimpleMetricEmbedding
from utils import train, evaluate, compute_representations

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False
# save_path = "./save_model/model.pth" # uncomment if you want to save the model
save_path = None

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "./data"
    ds_train = MNISTMetricDataset(mnist_download_root, split='train', remove_class=None)
    ds_train_no_0_class = MNISTMetricDataset(mnist_download_root, split='train', remove_class=0)
    unique_classes, num_classes = ds_train.get_unique_classes()
    # print("Unique Classes:", unique_classes, no_of_classes)
    ds_test = MNISTMetricDataset(mnist_download_root, split='test', remove_class=None)
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval', remove_class=None)

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader = DataLoader(ds_train, batch_size=64, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
    traineval_loader = DataLoader(ds_traineval, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

    emb_size = 32
    model = SimpleMetricEmbedding(1, emb_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        t0 = time.time_ns()
        train_loss = train(model, optimizer, train_loader, device, save_path)
        print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
        if EVAL_ON_TEST or EVAL_ON_TRAIN:
            print("Computing mean representations for evaluation...")
            representations = compute_representations(model, train_loader, num_classes, emb_size, device)
        if EVAL_ON_TRAIN:
            print("Evaluating on training set...")
            acc1 = evaluate(model, representations, traineval_loader, device)
            print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
        if EVAL_ON_TEST:
            print("Evaluating on test set...")
            acc1 = evaluate(model, representations, test_loader, device)
            print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
        t1 = time.time_ns()
        print(f"Epoch time (sec): {(t1-t0)/10**9:.1f}")
