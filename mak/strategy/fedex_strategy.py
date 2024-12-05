import copy
import flwr as fl
from logging import WARNING
from flwr.common.logger import log
from mak.utils.general import set_params
from torch.utils.data import DataLoader
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Tuple, Union, Optional
from mak.utils.fedex_utils import Hyperparameters
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from mak.utils.fedex_utils import  log_model_weights, log_hyper_config, log_hyper_params, discounted_mean
from numproto_updated import  ndarray_to_proto
from scipy.special import logsumexp
from numpy.linalg import norm


from flwr.common import (
    Scalar,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

def model_improved(results, weights):
    before_losses = np.array([res.metrics['before'] for _, res in results])
    after_losses = np.array([res.metrics['after'] for _, res in results])
    avg_before = np.sum(weights * before_losses)
    avg_after = np.sum(weights * after_losses)
    return (avg_after - avg_before) < 0

class FedEx(fl.server.strategy.FedAvg):
    """FedEx Strategy.
    Federated Hyperparameter Tuning: Challenges, Baselines, and Connections to Weight-Sharing
    https://proceedings.neurips.cc/paper/2021/hash/a0205b87490c847182672e8d371e9948-Abstract.html
    Implementation based on https://github.com/mkhodak/FedEx and https://github.com/ml-research/FEATHERS/tree/master/fedex_vanilla
    
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
        config,
        device = 'cpu',
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
        initial_params = [param.cpu().detach().numpy() for _, param in self.model.state_dict().items()]
        self.initial_parameters = self.last_weights = fl.common.ndarrays_to_parameters(initial_params)
        self.config = config['fedex']
        self.hyperparams = Hyperparameters(self.config['hyperparam_config_nr']) #hyperparmeter search space 
        self.hyperparams.save(self.config['hyperparam_file']) 
        self.config['dataset'] = config['common']['dataset']
        self.config['batch_size'] = config['client']['batch_size']
        

        self.log_distribution = np.full(len(self.hyperparams), -np.log(len(self.hyperparams)))
        self.distribution = np.exp(self.log_distribution)
        self.eta = np.sqrt(2*np.log(len(self.hyperparams)))
        self.discount_factor = self.config['discount_factor']
        self.use_gain_avg = self.config['use_gain_avg']
        self.distribution_history = []
        self.gain_history = [] # initialize with [0] to avoid nan-values in discounted mean
        self.log_gain_hist = []
       
    def __repr__(self) -> str:
        return "FedEx"
    

    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        # obtain client weights
        samples = np.array([fit_res[1].num_examples for fit_res in results])
        weights = samples / np.sum(samples)
        parameters_aggregated, _ = super().aggregate_fit(server_round, results, failures)

        # log current distribution
        self.distribution_history.append(self.distribution)
        dh = np.array(self.distribution_history)
        df = pd.DataFrame(dh)
        df.to_csv('distribution_history.csv')
        rh = np.array(self.log_gain_hist)
        df = pd.DataFrame(rh)
        df.to_csv('gain_history.csv')

        gains = self.compute_gains(weights, results)
        self.update_distribution(gains, weights)
        
        # sample hyperparameters and append them to the parameters
        serialized_dist = ndarray_to_proto(self.distribution)
        parameters_aggregated.tensors.append(serialized_dist.ndarray)

        # # log last hyperparam-configuration
        # for _, res in results:
        #     hidx = res.metrics['hidx']
        #     config = self.hyperparams[hidx]
        #     log_hyper_config(config, server_round, self.writer)
        # self.writer.add_histogram('gains', gains, server_round)
        
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
    def _sample_hyperparams(self):
        # obtain new learning rate for this batch
        distribution = torch.distributions.Categorical(torch.FloatTensor(self.distribution))
        hyp_idx = distribution.sample().item()
        hyp_config = self.hyperparams[hyp_idx]
        return hyp_config, hyp_idx
    
    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):
        """
        Initialize the model before training starts. Initialization sneds weights of initial_net
        passed in constructor

        Args:
            client_manager (fl.server.client_manager.ClientManager): Client Manager

        Returns:
            _type_: Initial model weights, distribution and hyperparameter configurations.
        """
        serialized_dist = ndarray_to_proto(self.distribution)
        self.initial_parameters.tensors.append(serialized_dist.ndarray)
        return self.initial_parameters
    
    def compute_gains(self, weights, results):
        """
        Computes the average gains/progress the model made during the last fit-call.
        Each client computes its validation loss before and after a backpop-step.
        The difference before - after is averaged and we compute (avg_before - avg_after) - gain_history.
        The gain_history is a discounted mean telling us how much gain we have obtained in the last
        rounds. If we obtain a better gain than in history, we will emphasize the corresponding
        hyperparameter-configurations in the distribution, if not these configurations get less
        likely to be sampled in the next round.

        Args:
            weights (_type_): Client weights
            results (_type_): Client results

        Returns:
            _type_: Gains
        """
        after_losses = [res.metrics['after'] for _, res in results]
        before_losses = [res.metrics['before'] for _, res in results]
        hidxs = [res.metrics['hidx'] for _, res in results]
        # compute (avg_before - avg_after)
        avg_gains = np.array([w * (a - b) for w, a, b in zip(weights, after_losses, before_losses)]).sum()
        self.gain_history.append(avg_gains)
        gains = []
        # use gain-history to obtain how much we improved on "average" in history
        baseline = discounted_mean(np.array(self.gain_history), self.discount_factor) if len(self.gain_history) > 0 else 0.0
        for hidx, al, bl, w in zip(hidxs, after_losses, before_losses, weights):
            gain = w * ((al - bl) - baseline) if self.use_gain_avg else w * (al - bl)
            client_gains = np.zeros(len(self.hyperparams))
            client_gains[hidx] = gain
            gains.append(client_gains)
        gains = np.array(gains)
        gains = gains.sum(axis=0)
        self.log_gain_hist.append(gains)
        return gains
    
    def update_distribution(self, gains, weights):
        """
        Update the distribution over the hyperparameter-search space.
        First, an exponantiated "gradient" update is made based on the gains we obtained.
        As a following step, we bound the maximum probability to be epsilon.
        Those configurations which have probability > epsilon after the exponantiated gradient step,
        are re-weighted such that near configurations are emphasized as well.
        NOTE: This re-weighting constraints our hyperparameter-search space to parameters on which an order can be defined.

        Args:
            gains (_type_): Gains obtained in last round
            weights (_type_): Weights of clients
        """
        denom = 1.0 if np.all(gains == 0.0) else norm(gains, float('inf'))
        self.log_distribution -= self.eta / denom * gains
        self.log_distribution -= logsumexp(self.log_distribution)
        self.distribution = np.exp(self.log_distribution)
    
