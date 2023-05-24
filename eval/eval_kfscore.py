import sys
sys.path.append(".")

import numpy as np
import torch
import matplotlib.pyplot as plt

from utility.dataset import KeyframeDataset
from utility.config import Config

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/dataset.json")

    # dataset
    # dataset = KeyframeDataset(train=False, config=config)
    features = torch.from_numpy(np.load("test.npy"))
    
    # analysis
    # features = dataset.features
    B, T, D = features.shape
    score = features[:, config.context_frames:-1, -1]
    # score = features[:, config.context_frames:-1, -1]

    # 
    # large_score = score > 0.8
    # score[~large_score] = 0
    
    # mean and standard deviation
    mean = torch.mean(score, dim=0)

    # plot
    categories = [f"{i}" for i in range(1, T-config.context_frames)]
    plt.figure(figsize=(10, 5))
    plt.bar(categories, mean)

    plt.title("Keyframe Score")
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.show()