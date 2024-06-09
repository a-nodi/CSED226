import umap
import wandb
import sklearn
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def load_data(data_path):    
    data = np.loadtxt(data_path, delimiter=',', skiprows=1, dtype=np.float32)
    return data[:, 1:]

def put_submission_data(prediction, file_name):
    id_array = np.arange(0, 5455 + 1).reshape(-1, 1)
    content = np.hstack((id_array, prediction.reshape(-1, 1)))
    df = pd.DataFrame(content, columns=['ID', 'Label'])
    df.to_csv(file_name, index=False)

def remap_index(prediction):
    _, idx = np.unique(prediction, return_index=True)
    list_of_index = prediction[np.sort(idx)]
    mapping = {list_of_index[i]: i for i in range(len(list_of_index))}

    return np.vectorize(mapping.__getitem__)(prediction)

def main():
    global args
    
    data_path = args.data_path
    data = load_data(data_path)

    run = wandb.init()
    
    print("UMAP running...")
    clusterable_embedding = umap.UMAP(
        n_neighbors=run.config.n_neighbors,
        min_dist=run.config.min_dist,
        n_components=run.config.n_components,
    ).fit_transform(data)
    
    clusterer = SpectralClustering(
        n_clusters=run.config.n_clusters,
        eigen_solver=run.config.eigen_solver,
        n_init=run.config.n_init,
        gamma=run.config.gamma,
        affinity=run.config.affinity,
        assign_labels=run.config.assign_labels,
        degree=run.config.degree,
        coef0=run.config.coef0,
        n_neighbors=run.config.n_neighbors,
    )
    
    print("Clustering running...")
    prediction = clusterer.fit_predict(clusterable_embedding)
    prediction = remap_index(prediction)
    
    print("Scoring...")
    if len(np.unique(prediction)) == 1:
        score = -1
    else:
        score = silhouette_score(data, prediction)
    
    wandb.log({'silhouette_score': score})
    
    fig = plt.figure()
    
    print("printing chart...")
    clusterable_embedding = umap.UMAP(
        n_neighbors=run.config.n_neighbors,
        min_dist=run.config.min_dist,
        n_components=3,
    ).fit_transform(data)
    
    fig.add_subplot(projection='3d').scatter(
        clusterable_embedding[:, 0], 
        clusterable_embedding[:, 1], 
        clusterable_embedding[:, 2],
        c=prediction, s=0.1, cmap='Spectral')
    
    wandb.log({'label_num': len(np.unique(prediction))})
    
    wandb.log({'chart' : wandb.Image(fig)})
    print("Done!")
    
def sweep():
    parser = argparse.ArgumentParser()
    # Configs
    parser.add_argument('--data_path', dest='data_path', type=str, default="train.csv")
    # parser.add_argument('--submission_data_path', dest='submission_data_path', type=str, default="submission.csv")
    parser.add_argument('--sweep_count', dest='sweep_count', type=int, default=300)
    # parser.add_argument('--test_size', dest='test_size', type=float, default=0.2)
    global args
    args = parser.parse_args()
    # WandB sweep
    sweep_config = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {
            'goal': 'maximize', 
            'name': 'silhouette_score'
        },
        'parameters': {
            'n_clusters': {
                'values': [*range(3, 20+ 1)]
            },
            'eigen_solver': {
                'values': ['arpack', 'lobpcg', None]
            },
            'n_init': {
                'values': [*range(1, 20 + 1)]
            },
            'gamma': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 2.0,
            },
            'affinity': {
                'values': ['nearest_neighbors', 'rbf']
            },
            'assign_labels': {
                'values': ['kmeans', 'discretize', 'cluster_qr']
            },
            'degree': {
                'values': [*range(1, 5 + 1)]
            },
            'coef0': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 1.0,
            },
            'n_neighbors': {
                'values': [*range(1, 1000 + 1)]
            },
            'min_dist': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 0.5,
            },
            'n_components': {
                'values': [*range(2, 24 + 1)]
            },
        },
    }
    
    sweep_id = wandb.sweep(sweep_config, project="IDA-HW6-cluster-spectral-clustering") 
    wandb.agent(sweep_id, function=main, count=args.sweep_count)
    
if __name__ == '__main__':
    sweep()