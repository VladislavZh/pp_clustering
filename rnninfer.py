import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from kmeans_pytorch import kmeans, kmeans_predict
from models.aemodel import RNNModel
from utils.metrics import info_score, purity


def save_embeddings(autoencoder, datatensor: TorchTensor, dir: str):
    """
    Obtain encodings from autoencoder for datatensor and save it to dir
    """
    Path(dir).mkdir(parents=True, exist_ok=True)

    return


if __name__ == "__main__":

    model = RNNModel(
        num_emb=1,
        vocab_size=10,
        input_dim=2,
        emb_dim=128,
        hidden_dim=10,
        encoder_dim=50,
        layers=1,
    )
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        model = model.cuda()
    print(model)

    batch_size = 16
    # load data
    datatensor = torch.load("data/pickledtensor.py")
    dataname = datatensor.split("/")[-1]
    dataname = datatensor.split(".")[0]
    pickleddata = TensorDataset(datatensor, datatensor)
    testloader = DataLoader(pickleddata, shuffle=True, batch_size=batch_size)
    # load model
    ckpt_dict = torch.load("checkpoints/default.ckpt")
    model.load_state_dict(ckpt_dict)
    # hidden1, hidden2 = model.init_hidden(batch_size, train_on_gpu)
    model.eval()
    # obtain embeddings
    event_count = {}
    for i, (inputs, _) in enumerate(testloader):
        if train_on_gpu:
            inputs = inputs.cuda()
        if len(inputs) != batch_size:
            break

        hidden1 = tuple([each.data for each in hidden1])
        hidden2 = tuple([each.data for each in hidden2])

        cat_targets = inputs[:, :, 0].unsqueeze_(-1)
        output, counts = torch.unique(cat_targets, return_counts=True)
        # freq-s of events
        for i in range(len(output)):
            event = output[i].item()
            if event not in event_count.keys():
                event_count[event] = counts[i].item()
            else:
                event_count[event] += counts[i].item()
        # embedding concat
        output_code = model.encode(inputs, hidden1, hidden2)
        if i == 0:
            embedtensor = output_code
        else:
            embedtensor = torch.cat(embedtensor, output_code)

    # save embeddings
    embedname = dataname + "_embed.pt"
    Path("embeddings").mkdir(parents=True, exist_ok=True)
    torch.save(embedtensor, os.path.join("embeddings", embedname))

    # get kmeans clusters
    cluster_ids_x, cluster_centers = kmeans(
        X=embedtensor,
        num_clusters=len(event_count.keys()),
        distance="euclidean",
        device=torch.device("cuda:0"),
    )
    print(cluster_ids_x)
    print(cluster_centers)
