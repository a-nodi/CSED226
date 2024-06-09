import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm
def cluster(data, k, eps):
    """
    kmeans = KMeans(
        n_clusters=k, 
        random_state=20220923, 
        n_init="auto",
        algorithm="lloyd",
        max_iter=100000,
        ).fit(data)
    
    return kmeans.labels_
    """
    cluster = DBSCAN(
        eps=2.973106622684774, 
        min_samples=42,
        metric='chebyshev',
        algorithm='auto',
        leaf_size=24,
        n_jobs=-1,
        ).fit(data)
    return cluster.labels_
    

def load_data(data_path):    
    data = np.loadtxt(data_path, delimiter=',', skiprows=1, dtype=np.float32)
    return data[:, 1:]

def split_feature_label(data):
    
    feature = data[:, :-1]
    
    label = data[:, -1]
    label = label.reshape(-1, 1)
    
    return feature, label

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
    data_path = 'train.csv'
    data = load_data(data_path)
    # for i in range(2, 20):
    max_score = 0
    max_eps = 0

    # for i in tqdm(range(10, 300)):
    """
        eps = 0.01 * i
        prediction = cluster(data, 0, eps)
        prediction = remap_index(prediction)
        # print(np.unique(prediction, return_counts=True))
        score = silhouette_score(data, prediction)
        if score > max_score:
            max_score = score
            max_eps = eps
        print(f" [LOG] eps = {eps :.2f} score = {score :.4f} max_eps =  {max_eps :.2f} max_score = {max_score :.4f}")
    """
    prediction = cluster(data, 0, max_eps)
    prediction = remap_index(prediction)
    put_submission_data(prediction, f'cluster_submission_dbscan_max_score_with_minsample_contraint.csv')

if __name__ == "__main__":
    main()