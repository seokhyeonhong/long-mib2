import os
import pickle
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pymovis.motion import BVH
from pymovis.ops import rotation, motionops
from pymovis.utils import util

from utility.dataset import KeyframeDataset
from utility.config import Config
from utility import utils

def main():
    config = Config.load("configs/spnet.json")
    dset = KeyframeDataset(train=False, config=config)
    dloader = DataLoader(dset, batch_size=64, shuffle=False)

    results = []
    for d in dloader:
        B, T, D = d.shape
        local_R6, root_p, score = torch.split(d, [D-4, 3, 1], dim=-1)
        local_R6, global_p = utils.get_motion(torch.cat([local_R6, root_p], dim=-1), dset.skeleton)
        local_R6 = local_R6.reshape(B, T, -1)
        global_p = global_p.reshape(B, T, -1)
        features = torch.cat([local_R6, global_p, score], dim=-1)
        results.append(features.cpu().numpy())

    results = np.concatenate(results, axis=0)
    print(results.shape)
    np.save("dataset/test.npy", results)


if __name__ == "__main__":
    main()