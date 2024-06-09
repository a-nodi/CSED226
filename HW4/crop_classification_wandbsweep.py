import wandb
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler

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
    
def predict_submission_data():
    pass

def main():
    # WandB
    global args
    run = wandb.init()
    scaler = StandardScaler()
    
    
    feature_label = split_feature_label(load_data(args.train_data_path)), 
    train_feature, train_label = feature_label[0]
    train_feature, train_label, test_feature, test_label = split_data(
        train_feature,
        train_label,
        args.test_size
    )
    
    train_label, test_label = train_label.astype(np.int32), test_label.astype(np.int32)
    
    # Preprocessing
    # train_feature = scaler.fit_transform(train_feature)
    # test_feature = scaler.fit_transform(test_feature)
    
    
    submission_feature = load_data(args.submission_data_path)
    
    decision_tree = DecisionTreeClassifier(
        criterion=wandb.config.criterion,
        splitter=wandb.config.splitter,
        max_depth=wandb.config.max_depth,
        min_samples_split=wandb.config.min_samples_split,
        min_samples_leaf=wandb.config.min_samples_leaf,
        min_weight_fraction_leaf=wandb.config.min_weight_fraction_leaf,
        max_features=wandb.config.max_features,
        random_state=wandb.config.random_state,
        max_leaf_nodes=wandb.config.max_leaf_nodes,
        min_impurity_decrease=wandb.config.min_impurity_decrease,
        class_weight=wandb.config.class_weight,
        ccp_alpha=wandb.config.ccp_alpha
    )
    
    decision_tree.fit(train_feature, train_label)
    
    test_prediction = decision_tree.predict(test_feature).astype(np.int32)
    test_f1_score = f1_score(test_label, test_prediction, average='macro')
    wandb.log({'f1_score': test_f1_score})
    submission_prediction = decision_tree.predict(submission_feature)
    
    
    pass

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Configs
    parser.add_argument('--train_data_path', dest='train_data_path', type=str, default="crop_train.csv")
    parser.add_argument('--submission_data_path', dest='submission_data_path', type=str, default="crop_test.csv")
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
            'criterion': {'values': ['gini', 'entropy', 'log_loss']},
            'splitter': {'values': ['best', 'random']},
            'max_depth': {'values': [*range(1, 20)]},
            'min_samples_split': {'values': [*range(1, 100)]},
            'min_samples_leaf': {'values': [*range(1, 100)]},
            'min_weight_fraction_leaf': {'min': 0.0, 'max': 0.5},
            'max_features': {'values': ['sqrt', 'log2', None]},
            'random_state': {'values': [20220923]},
            'max_leaf_nodes': {'values': [*range(1, 100)]},
            'min_impurity_decrease': {'min': 0.0, 'max': 0.5},
            'class_weight': {'values': [None, 'balanced']},
            'ccp_alpha': {'min': 0.0, 'max': 0.5},
        },
    }
    
    sweep_id = wandb.sweep(sweep_config, project="IDA-HW4-crop") 
    wandb.agent(sweep_id, function=main, count=args.sweep_count)
    
    # main()