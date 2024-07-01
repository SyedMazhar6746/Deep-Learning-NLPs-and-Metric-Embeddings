#!/usr/bin/python3

import os
import time
import torch.optim
from Task_1_mnist_data import MNISTMetricDataset
from torch.utils.data import DataLoader
from Task_2_metric import SimpleMetricEmbedding
from utils import train, evaluate, compute_representations
from matplotlib import pyplot as plt
import numpy as np

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False
# save_path = "./save_model/model.pth" # uncomment if you want to save the model
save_path = "./save_model/model_no_0_class.pth" # uncomment if you want to save the model
# save_path = None

def get_colormap():
    # Cityscapes colormap for first 10 classes
    colormap = np.zeros((10, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    return colormap

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    emb_size = 32
    model = SimpleMetricEmbedding(1, emb_size).to(device)

    # Load the trained parameters
    model.load_state_dict(torch.load(save_path))

    # CHANGE ACCORDING TO YOUR PREFERENCE
    colormap = get_colormap()
    mnist_download_root = "./data"
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    
    X = ds_test.images
    Y = ds_test.targets

    print("Fitting PCA directly from images...")
    test_img_rep2d = torch.pca_lowrank(ds_test.images.view(-1, 28 * 28), 2)[0]
    plt.scatter(test_img_rep2d[:, 0], test_img_rep2d[:, 1], color=colormap[Y[:]] / 255., s=5)
    plt.show()
    plt.figure()

    
    print("Fitting PCA from feature representation")
    with torch.no_grad():
        model.eval()

        # Move input tensor to the same device as the model
        X = X.to(device)

        test_rep = model.get_features(X.unsqueeze(1))
        test_rep2d = torch.pca_lowrank(test_rep, 2)[0]

        # Move the tensor to CPU before plotting
        test_rep2d = test_rep2d.cpu().numpy()

        plt.scatter(test_rep2d[:, 0], test_rep2d[:, 1], color=colormap[Y[:]] / 255., s=5)
        plt.show()
