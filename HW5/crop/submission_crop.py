
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def load_data(data_path):    
    data = np.loadtxt(data_path, delimiter=',', skiprows=1, dtype=np.float32)
    return data

def split_feature_label(data):
    
    feature = data[:, :-1]
    
    label = data[:, -1]
    label = label.reshape(-1, 1)
    
    return feature, label

def split_data(feature, label, test_size):
    train_feature, test_feature, train_label, test_label = train_test_split(
        feature,
        label, 
        test_size=test_size, 
        random_state=20220923,
        stratify=label
    )
    
    return train_feature, train_label, test_feature, test_label
    
def put_submission_data(prediction):
    id_array = np.arange(0, 3000).reshape(-1, 1)
    content = np.hstack((id_array, prediction.reshape(-1, 1)))
    df = pd.DataFrame(content, columns=['ID', 'Age'])
    df.to_csv('/home/anodi/code/crop/submission.csv', index=False)
    
def main():
    # WandB
    global args
    # run = wandb.init()

    feature_label = split_feature_label(load_data(args.train_data_path)), 
    train_feature, train_label = feature_label[0]
    train_feature, train_label, test_feature, test_label = split_data(
        train_feature,
        train_label,
        args.test_size
    )
    
    train_label, test_label = train_label.astype(np.int32), test_label.astype(np.int32)
    
    submission_feature = load_data(args.submission_data_path)
    
    xgboost_classifier = XGBClassifier(
        booster='gbtree',
        silent=True,
        nthread=4,
        objective='multi:softmax',
        random_state=20220923,
        colsample_bylevel=0.9, 
        colsample_bytree=0.9,
        gamma=0.1,
        max_depth=8,
        min_child_weight=80,
        n_estimators=100
    )
    
    xgboost_classifier.fit(np.vstack((train_feature, test_feature)), np.vstack((train_label, test_label)))
    submission_prediction = xgboost_classifier.predict(submission_feature)
    put_submission_data(submission_prediction)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Configs
    parser.add_argument('--train_data_path', dest='train_data_path', type=str, default="/home/anodi/code/crop/crop_train.csv")
    parser.add_argument('--submission_data_path', dest='submission_data_path', type=str, default="/home/anodi/code/crop/crop_test.csv")
    parser.add_argument('--sweep_count', dest='sweep_count', type=int, default=10000)
    parser.add_argument('--test_size', dest='test_size', type=float, default=0.2)
    
    # Parse args
    global args
    args = parser.parse_args()
    
    # sweep_id = wandb.sweep(sweep_config, project="IDA-HW4-crop") 
    # wandb.agent(sweep_id, function=main, count=args.sweep_count)
    
    main()