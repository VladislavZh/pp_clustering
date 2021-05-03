import matplotlib.pyplot as plt
import numpy as np
import torch
from kmeans_pytorch import kmeans, kmeans_predict

# set random seed
np.random.seed(123)
# data
data_size, dims, num_clusters = 1000, 2, 3
x = np.random.randn(data_size, dims) / 6
x = torch.from_numpy(x)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# k-means
cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean', device=device
)
# cluster IDs and cluster centers
print(cluster_ids_x)
print(cluster_centers)

# predict cluster ids for y
cluster_ids_y = kmeans_predict(
    y, cluster_centers, 'euclidean', device=device
)
