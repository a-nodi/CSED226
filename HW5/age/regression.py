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
    data["weight5"] = data["weight1"] - (data["weight2"] + data["weight3"] + data["weight4"])
    data["weight_ratio"] = data["weight1"] / (data["weight1"] - data["weight5"])
    # data["weight_diff_score"] = (data["weight5"] / data["weight1"]) * 100
    if 'age' in data.columns:
        data = data[['gender_F', 'gender_I', 'gender_M', 'length', 'diameter', 'height', 'weight1', 'weight2', 'weight3', 'weight4', 'weight5', 'weight_ratio', 'age']]
    else:
        data = data[['gender_F', 'gender_I', 'gender_M', 'length', 'diameter', 'height', 'weight1', 'weight2', 'weight3', 'weight4', 'weight5', 'weight_ratio', ]]
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
        random_state=20220923
    )
    
    return train_feature, train_label, test_feature, test_label
    
def put_submission_data(prediction):
    id_array = np.arange(0, 1000).reshape(-1, 1)
    content = np.hstack((id_array, prediction.reshape(-1, 1)))
    df = pd.DataFrame(content, columns=['ID', 'Age'])
    df.to_csv('submission.csv', index=False)
    
    
def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return(np.sqrt(mse))


def main():
    # WandB
    global args
    # run = wandb.init()
    # kfold = KFold(n_splits=5)
    
    feature_label = split_feature_label(load_data(args.train_data_path)), 
    train_feature, train_label = feature_label[0]
    train_feature, train_label, test_feature, test_label = split_data(
        train_feature,
        train_label,
        args.test_size
    )
    
    train_label, test_label = train_label.astype(np.int32), test_label.astype(np.int32)
        
    submission_feature = load_data(args.submission_data_path)
    
    regressor = SVR()
    
    rmse_score = make_scorer(rmse, greater_is_better=False)
    
    grid_regressor =GridSearchCV(
        regressor,
        param_grid={  # param_grid
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [*range(4, 6 + 1)],
            'gamma': np.linspace(0.4, 0.6, 5),
            'coef0': np.linspace(0.7, 0.9, 5),
            'tol': np.linspace(3e-3, 5e-3, 3),
            'C': np.linspace(0.8, 1.2, 5),
            'epsilon': np.linspace(0.7, 0.8, 4),
            'shrinking': [True, False]
        },
        cv=3,
        refit=True,
        verbose=100,
        scoring = rmse_score,
        # n_jobs= 8,
        # return_train_score=True,
        # n_iter=10000
    )
    
    grid_regressor.fit(train_feature, train_label.ravel())
    
    test_prediction = grid_regressor.predict(test_feature)
    test_rmse_score = mean_squared_error(test_label.ravel(), np.around(test_prediction).astype(np.int32).ravel()) ** 0.5
    # wandb.log({'RMSE': test_rmse_score})
    print(grid_regressor.best_params_)
    print(test_rmse_score)
    submission_prediction = grid_regressor.predict(submission_feature)
    print(submission_prediction)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Configs
    parser.add_argument('--train_data_path', dest='train_data_path', type=str, default="age_train.csv")
    parser.add_argument('--submission_data_path', dest='submission_data_path', type=str, default="age_test.csv")
    parser.add_argument('--sweep_count', dest='sweep_count', type=int, default=5000)
    parser.add_argument('--test_size', dest='test_size', type=float, default=0.2)
    
    # Parse args
    global args
    args = parser.parse_args()
    
    # WandB sweep
    sweep_config = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize', 
            'name': 'RMSE'
        },
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        'parameters': {
            'kernel': {'values': ['linear', 'poly', 'rbf', 'sigmoid']},
            'degree': {'values': [*range(1, 5 + 1)]},
            'gamma': {'min': 0.001, 'max': 1.0},
            'coef0': {'min': 0.0, 'max': 1.0},
            'tol': {'min': 1e-4, 'max': 1e-2},
            'C': {'min': 0.1, 'max': 1.0},
            'epsilon': {'min': 0.01, 'max': 1.0},
            'shrinking': {'values': [True, False]}
        },
    }
    
    # sweep_id = wandb.sweep(sweep_config, project="IDA-HW5-crop") 
    # wandb.agent(sweep_id, function=main, count=args.sweep_count)
    
    main()