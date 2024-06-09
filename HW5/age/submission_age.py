# import wandb
import sklearn
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV

def load_data(data_path):    
    
    # data = np.loadtxt(data_path, delimiter=',', skiprows=1, dtype=np.float32)
    data = pd.read_csv(data_path)
    data = pd.get_dummies(data, columns=['gender'])
    if 'age' in data.columns:
        data = data[['gender_F', 'gender_I', 'gender_M', 'length', 'diameter', 'height', 'weight1', 'weight2', 'weight3', 'weight4', 'age']]
    else:
        data = data[['gender_F', 'gender_I', 'gender_M', 'length', 'diameter', 'height', 'weight1', 'weight2', 'weight3', 'weight4']]
    data = data.to_numpy()
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
        
    )
    
    return train_feature, train_label, test_feature, test_label
    
def put_submission_data(prediction):
    id_array = np.arange(0, 1000).reshape(-1, 1)
    content = np.hstack((id_array, prediction.reshape(-1, 1)))
    df = pd.DataFrame(content, columns=['ID', 'Age'])
    df.to_csv('/home/anodi/code/age/submission.csv', index=False)

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
    
    # Preprocessing
    """
    if wandb.config.scale == "Normalizer":
        scaler = Normalizer()
        train_feature = scaler.fit_transform(train_feature)
        test_feature = scaler.fit_transform(test_feature)
    
    if wandb.config.scale == "Standard Scaler":
        scaler = StandardScaler()
        train_feature = scaler.fit_transform(train_feature)
        test_feature = scaler.fit_transform(test_feature)
        
    if wandb.config.scale == "MinMax Scaler":
        scaler = MinMaxScaler()
        train_feature = scaler.fit_transform(train_feature)
        test_feature = scaler.fit_transform(test_feature)
        
    if wandb.config.scale == "Robust Scaler":
        scaler = RobustScaler()
        train_feature = scaler.fit_transform(train_feature)
        test_feature = scaler.fit_transform(test_feature)
    """
        
    submission_feature = load_data(args.submission_data_path)
    
    regressor = SVR(
        C=1.0,
        coef0=0.8,
        degree=5,
        epsilon=0.775,
        gamma=0.5,
        kernel='poly',
        shrinking=True,
        tol=0.003
    )
    
    regressor.fit(np.vstack((train_feature, test_feature)), np.vstack((train_label, test_label)).ravel())
    
    test_prediction = regressor.predict(test_feature)
    test_rmse_score = mean_squared_error(test_label, np.around(test_prediction).astype(np.int32)) ** 0.5
    # wandb.log({'RMSE': test_rmse_score})
    submission_prediction = regressor.predict(submission_feature)

    put_submission_data(np.around(submission_prediction).astype(np.int32))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Configs
    parser.add_argument('--train_data_path', dest='train_data_path', type=str, default="/home/anodi/code/age/age_train.csv")
    parser.add_argument('--submission_data_path', dest='submission_data_path', type=str, default="/home/anodi/code/age/age_test.csv")
    parser.add_argument('--sweep_count', dest='sweep_count', type=int, default=10000)
    parser.add_argument('--test_size', dest='test_size', type=float, default=0.2)
    
    # Parse args
    global args
    args = parser.parse_args()
    
    # sweep_id = wandb.sweep(sweep_config, project="IDA-HW4-crop") 
    # wandb.agent(sweep_id, function=main, count=args.sweep_count)
    
    main()