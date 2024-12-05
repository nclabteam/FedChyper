import numpy as np
import pandas as pd
import os
import json

class Hyperparameters:
    def __init__(self, nr_configs) -> None: #nr_configs : size of hyperparameter search space
        self.sample_hyperparams = lambda: {
            'learning_rate': 10 ** (np.random.uniform(-4, -1)),  # learning_rate is set to a random uniform value in range of 0.1 and 0.0001
            # 'weight_decay': 10.0 ** np.random.uniform(low=-5.0, high=-1.0), # random value between 0.00001 and 0.1.
            # 'momentum': np.random.uniform(low=0.0, high=1.0), # random value between 0.0 and 1.0
            # 'dropout': np.random.uniform(low=0, high=0.2), #random value between 0 and 0.2
            #'arch_learning_rate': 10 ** (np.random.uniform(-5, -2)), # for search-phase
            #'arch_weight_decay': 10.0 ** np.random.uniform(low=-5.0, high=-1.0), # for search-phase
        }
     
        self.hyperparams = [self.sample_hyperparams() for _ in range(nr_configs)] #returns a hyperparameter seach space of size nr_configs.

    def read_from_csv(self, file):
        df = pd.read_csv(file, index_col=0)
        arr = []
        for _, row in df.iterrows():
            arr.append(row.to_dict())
        self.hyperparams = arr

    def save(self, file):
        df = pd.DataFrame.from_dict(self.to_dict())
        dir = file.split('/')[1]
        if not os.path.exists(dir):
            os.mkdir(dir)
        df.to_csv(file)

    def __getitem__(self, idx):
        return self.hyperparams[idx]

    def __len__(self):
        return len(self.hyperparams)

    def to_dict(self):
        hyperparam_dict = {}
        for config in self.hyperparams:
            for key, val in config.items():
                if key in hyperparam_dict.keys():
                    hyperparam_dict[key].append(val)
                else:
                    hyperparam_dict[key] = [val]
        return hyperparam_dict
    


def discounted_mean(series, gamma=1.0):
    weight = gamma ** np.flip(np.arange(len(series)), axis=0)
    return np.inner(series, weight) / weight.sum()

def log_model_weights(model, step, writer):
    for name, weight in model.named_parameters():
        writer.add_histogram(name, weight, step)
    writer.flush()

def log_hyper_config(config, step, writer):
    for key, hyperparam in config.items():
        writer.add_scalar(key, hyperparam, step)

def log_hyper_params(hyper_param_dict):
    to_be_persisted = {k: list(v) for k, v in hyper_param_dict.items()}
    with open('hyperparameters.json', 'w') as f:
        json.dump(to_be_persisted, f)
  

def get_hyperparameter_id(name, client_id):
    # hyperparameter-names must have format arbitrary_name_[round_number]
    # thus we cut off "_[round_number]" and add "client_[id]_" to obtain unique
    # log-id for each client such that each hyper-parameter configuration is 
    # logged in one time-diagram per client
    split_name = name.split('_')
    split_name = split_name[:-1]
    log_name = '_'.join(split_name)
    log_name = 'client_{}_'.format(client_id) + log_name
    return log_name

def prepare_log_dirs():
    if not os.path.exists('./hyperparam-logs/'):
        os.mkdir('./hyperparam-logs')
    if not os.path.exists('./models/'):
        os.mkdir('./models')

class ProtobufNumpyArray:
    """
        Class needed to deserialize numpy-arrays coming from flower
    """
    def __init__(self, bytes) -> None:
        self.ndarray = bytes
