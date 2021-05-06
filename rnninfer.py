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


if __name__ == "__main__":
    
    vocab_size = 10
    model = RNNModel(
        num_emb=1,
        vocab_size=vocab_size,
        input_dim=2,
        emb_dim=128,
        hidden_dim=50,
        encoder_dim=10,
        n_layers=1,
    )
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        model = model.cuda()
    print(model)

    # load data
    datapath = "data/atm_train_day_seqlen300.pt"
    datatensor = torch.load(datapath)
    dataname = datapath.split("/")[-1]
    dataname = dataname.split(".")[0]
    pickleddata = TensorDataset(datatensor, datatensor)
    batch_size = 1
    testloader = DataLoader(pickleddata, shuffle=False, batch_size=batch_size)
    # load model
    ckpt_dict = torch.load("checkpoints/atm_train_day_seqlen300-checkpoint-20.pth")
    model.load_state_dict(ckpt_dict)
    hidden1, hidden2 = model.init_hidden(batch_size, train_on_gpu)
    model.eval()
    # obtain embeddings
    for i, (inputs, _) in enumerate(testloader):
        if train_on_gpu:
            inputs = inputs.cuda()

        hidden1 = tuple([each.data for each in hidden1])
        hidden2 = tuple([each.data for each in hidden2])

        # embedding concat
        output_code = model.encode(inputs, hidden1, hidden2)
        if i == 0:
            embedtensor = output_code.unsqueeze_(-1)
        else:
            embedtensor = torch.cat((embedtensor, output_code.unsqueeze_(-1)), -1)

    # save embeddings
    embedname = dataname + "_embed.pt"
    embedtensor = embedtensor.permute(2, 0, 1)
    Path("embeddings").mkdir(parents=True, exist_ok=True)
    torch.save(embedtensor, os.path.join("embeddings", embedname))
    embedtensor = embedtensor.mean(1)
    print(embedtensor.shape)
    # get kmeans clusters
    cluster_ids_x, cluster_centers = kmeans(
        X=embedtensor,
        num_clusters=vocab_size,
        distance="euclidean",
        device=torch.device("cuda:0"),
    )
    # obtain gt ids and learned ids
    assert batch_size == 1, "as we iterate over each user id consequently"
    gt_ids = []
    learned_ids = []
    for i, (inputs, _) in enumerate(testloader):
        if train_on_gpu:
            inputs = inputs.cuda()
            # column of events history for user id
            cat_targets = inputs.squeeze(0)[:, 0]
            gt_ids.append(torch.mode(cat_targets).values.item())
    # smpl heuristics - learned id is most frequent among cluster users
    for i in range(len(cluster_ids_x)):
        sameclusterusers = [j for j,x in enumerate(cluster_ids_x) if x == cluster_ids_x[i]]
        cat_targets = datatensor[sameclusterusers]
        cat_targets = cat_targets[:,:, 0].flatten()
        learned_ids.append(torch.mode(cat_targets).values.item())
    

    # calculate metrics
    gt_ids = torch.FloatTensor(gt_ids)
    learned_ids = torch.FloatTensor(learned_ids)
    #learned_ids = torch.zeros(gt_ids.shape)
    purity = purity(learned_ids, gt_ids)
    info_score = info_score(learned_ids, gt_ids, 10)
    print("\n data:", dataname)
    print("embed:", embedname)
    print("purity:", purity)
    #print("info_score:", info_score)
