from typing import Dict, List, Tuple, Union, Optional
from logging import WARNING
import flwr as fl
from dataclasses import dataclass, asdict
import json
from functools import reduce
import numpy as np
from math import exp
from logging import INFO
from flwr.common.logger import log

from flwr.common import (
    Scalar,
    FitRes,
    FitIns,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.common import NDArray, NDArrays
from mak.utils.general import set_params, test, filter_straggler_results
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

class FedAvgS(fl.server.strategy.FedAvg):

    """FedAvg Strategy with stragglers.
    Implementation based on https://github.com/adap/flower/blob/main/baselines/fedprox/fedprox/strategy.py
    """
     
    def __init__(
        self,
        model,
        test_data,
        fraction_fit: float,
        fraction_evaluate: float,
        min_fit_clients: int,
        min_evaluate_clients : int,
        min_available_clients : int,
        evaluate_fn,
        evaluate_metrics_aggregation_fn,
        apply_transforms,
        device = 'cpu',
        config = None,
        on_fit_config_fn = None,
        **kwargs
    ) -> None:
        super().__init__(fraction_fit=fraction_fit,
                         fraction_evaluate = fraction_evaluate,
                         min_fit_clients = min_fit_clients,
                         min_evaluate_clients = min_evaluate_clients,
                         min_available_clients = min_available_clients,
                         evaluate_fn = evaluate_fn,
                         on_fit_config_fn = on_fit_config_fn)
        self.model = model
        self.test_data = test_data
        self.fraction_fit = fraction_fit
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.apply_transforms = apply_transforms
        self.device = device
        self.config_sim = config

    def __repr__(self) -> str:
        return " FedAvg With Stragglers"

    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if self.config_sim['straggler']['enable']:
            results = filter_straggler_results(results=results)
            log(INFO,f" =>>>>> Round : {server_round} Non Straggler Clients : {len(results)}") 
        # call the parent `aggregate_fit()` (i.e. that in standard FedAvg)
        return super().aggregate_fit(server_round, results, failures)

        
    
        