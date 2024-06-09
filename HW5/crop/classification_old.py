
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# 1. [DONE] Load data
# 2. [DONE] Split data
#  2.1 [DONE] train, test split portion (default: 0.8, 0.2) set
# 3. [TODO] test data f1 score wandb log
# 4. [TODO] tree diagram wandb log
# 5. [TODO] train, val accuracy wandb log
# 6. [TODO] train, val, test f1 score wandb log
# 7. [TODO] submission data csv print
# 8. [TODO] feature importance wandb log (hist)

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
    df = pd.DataFrame(content, columns=['ID', 'Class'])
    df.to_csv('submission.csv', index=False)

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
        random_state=20220923 
    )
    
    grid_classifier = GridSearchCV(
        xgboost_classifier,
        param_grid={
            'min_child_weight': [(80 + 10 * i) for i in range(0, 4 + 1)],
            'max_depth': [*range(3, 8 + 1)],
            'gamma': np.linspace(0.1, 1.0, 5),
            'colsample_bytree': np.linspace(0.6, 0.9, 3),
            'colsample_bylevel': np.linspace(0.6, 0.9, 3),
            'n_estimators': [80, 100]
        },
        cv=3,
        refit=True,
        verbose=100,
        scoring ='f1_micro',
    
    )
    
    grid_classifier.fit(train_feature, train_label)
    
    test_prediction = grid_classifier.predict(test_feature).astype(np.int32)
    test_f1_score = f1_score(test_label, test_prediction, average='macro')
    # wandb.log({'f1_score': test_f1_score})
    print(grid_classifier.best_params_)
    print(test_f1_score)
    submission_prediction = xgboost_classifier.predict(submission_feature)
    print(submission_prediction)
    
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
    
    # WandB sweep
    sweep_config = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {
            'goal': 'maximize', 
            'name': 'f1_score'
        },
        'parameters': {
            'booster': {'values': ['gbtree']},
            'min_child_weight': {'values': [*range(1, 100 + 1)]},
            'max_depth': {'values': [*range(3, 10 + 1)]},
            'gamma': {'min': 0.0, 'max': 1.0},
            'colsample_bytree': {'min': 0.6, 'max': 0.9},
            'colsample_bylevel': {'min': 0.6, 'max': 0.9},
            'n_estimators': {'values': [*range(1, 100 + 1)]},
        },
    }
    
    # sweep_id = wandb.sweep(sweep_config, project="IDA-HW5-crop") 
    # wandb.agent(sweep_id, function=main, count=args.sweep_count)
    
    main()