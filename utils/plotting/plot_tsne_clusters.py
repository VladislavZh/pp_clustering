import json
import os
import sys
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("science")
sys.path.insert(1, '/pp_clustering')
from models.LSTM import LSTMMultiplePointProcesses
from utils.data_preprocessor import get_dataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

# Fig 6-7
data_name = "K4_C5"
data_path = os.path.join("../../data", data_name)
exper_path = os.path.join("../../experiments", data_name, "exp_0")
# obtain gt and cohortney labels
df = pd.read_csv(os.path.join(exper_path, "compareclusters.csv"))
coh_clusters = df["coh_cluster"].values.tolist()
gt_clusters = df["cluster_id"].values.tolist()

# load model to obtain embeddings
with open(os.path.join(exper_path, "args.json")) as json_file:
    config = json.load(json_file)
n_steps = config["n_steps"]
n_classes = config["n_classes"]
model_weights = os.path.join(exper_path, "last_model.pt")
# init model
model = LSTMMultiplePointProcesses(
    n_classes + 1,
    config["hidden_size"],
    config["num_layers"],
    n_classes,
    config["n_clusters"],
    n_steps,
    dropout=config["dropout"],
).to(config["device"])
model = torch.load(model_weights, map_location=torch.device(config["device"]))
model.eval()

# obtain embeddings from hidden states
data, target = get_dataset(data_path, model.num_classes, n_steps, col_to_select=None)
dataloader = DataLoader(data, batch_size=16, shuffle=False, num_workers=1)
embed_list = []
for batch_idx, sample in enumerate(dataloader):
    lambdas, hiddens, cells = model.forward(sample.to(config["device"]), return_states=True)
    embed_list.append(torch.stack(hiddens))

# transform embedding to tensor
embeddings = embed_list[0]
for i in range(1, len(embed_list)):
    embeddings = torch.cat((embeddings, embed_list[i]), dim=2)
embeddings = embeddings.permute(2,0,1,3)
embeddings = embeddings[:,:,-1,:]
embeddings = torch.flatten(embeddings, start_dim=1)

# tsne and visualization
embeddings = embeddings.cpu().detach().numpy()
t_embeddings = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(embeddings)
print(t_embeddings.shape)
t_embeddings = pd.DataFrame({'x': t_embeddings[:,0], 'y': t_embeddings[:,1]})
fig = plt.figure(figsize=(16,7))
num_of_colors = max(gt_clusters)+1
#fig, axs = plt.subplots(1, len(metrics), constrained_layout=True)
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x='x', y='y',
    hue=gt_clusters,
    palette=sns.color_palette("hls", num_of_colors),
    data=t_embeddings,
    legend="full",
    alpha=0.3
)
ax2 = plt.subplot(1, 2, 2)
num_of_colors = max(coh_clusters)+1
sns.scatterplot(
    x='x', y='y',
    hue=coh_clusters,
    palette=sns.color_palette("hls", num_of_colors),
    data=t_embeddings,
    legend="full",
    alpha=0.3
)
fig.savefig(data_name + "_tsne.pdf", dpi=400, bbox_inches="tight")
