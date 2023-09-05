# We use this file to calculate the accuracy of the clustering given by the model
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score , davies_bouldin_score , calinski_harabasz_score




def sample_nodes(emb, nbr_nodes: int = 30000 ):
    """
    emb: embedding matrix
    nbr_nodes: number of nodes to sample
    """
    assert nbr_nodes <= emb.shape[0]
    sampling_indexes = torch.randperm(emb.shape[0])[:nbr_nodes]
    sampled_data = emb.clone()[sampling_indexes]
    return sampled_data


def get_clusters(emb, k):
    """
    emb: embedding matrix
    k: number of clusters
    """
    kmeans = KMeans(n_clusters=k, random_state=0).fit(emb)
    return kmeans.labels_


def get_accuracy(emb, k):
    """
    emb: embedding matrix
    k: number of clusters
    """
    sampled_emb = sample_nodes(emb)
    clusters = get_clusters(sampled_emb, k)
    sil_score = silhouette_score(sampled_emb, clusters)
    db_score = davies_bouldin_score(sampled_emb, clusters)
    ch_score = calinski_harabasz_score(sampled_emb, clusters)
    return sil_score, db_score, ch_score

