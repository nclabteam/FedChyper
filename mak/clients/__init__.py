
import flwr as fl
from flwr_datasets import FederatedDataset
from mak.clients.fedavg_client import FedAvgClient
from mak.clients.fedprox_client import FedProxClient
from mak.clients.fedex_client import FedExClient
from flwr_datasets.partitioner import NaturalIdPartitioner
import numpy as np
import pandas as pd
import os

def get_client_fn(config_sim: dict, dataset: FederatedDataset, model, device, apply_transforms, save_path):
    strategy = config_sim['server']['strategy']
    client_class = get_client_class(strategy)

    # Defines a straggling schedule for each clients, i.e at which round will they
    # be a straggler. This is done so at each round the proportion of straggling
    # clients is respected
    straggler_prob = config_sim['straggler']['straggler_prob']
    stragglers_mat = np.transpose(
        np.random.choice(
            [0, 1], size=(config_sim['server']['num_rounds'], config_sim['server']['num_clients']), p=[1 - straggler_prob, straggler_prob]
        )
    )
    if config_sim['straggler']['enable']:
        stragglers_df = pd.DataFrame(stragglers_mat)
        stragglers_df.to_csv(os.path.join(save_path,'straggler_mat.csv'),index=False)


    def client_fn(cid: str) -> fl.client.Client:
        if isinstance(dataset, NaturalIdPartitioner):
            client_dataset = dataset.load_partition(int(cid))
        else:
            client_dataset = dataset.load_partition(int(cid), "train")
        client_dataset_splits = client_dataset.train_test_split(test_size=0.15)
        trainset = client_dataset_splits["train"].with_transform(apply_transforms)
        valset = client_dataset_splits["test"].with_transform(apply_transforms)
        return client_class(client_id = cid, model=model, trainset=trainset, valset=valset, config_sim = config_sim, device=device,straggler_schedule=stragglers_mat[int(cid)]).to_client()
    return client_fn

def get_client_class(strategy: str):
    if strategy == 'FedProx':
        return FedProxClient
    elif strategy == 'FedEx':
        return FedExClient
    else:
        return FedAvgClient
