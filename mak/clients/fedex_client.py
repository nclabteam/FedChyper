from torch.utils.data import DataLoader
from mak.utils.helper import get_optimizer
from mak.clients.base_client import BaseClient

from mak.utils.fedex_utils import Hyperparameters
import torch
from collections import OrderedDict
from mak.utils.general import set_params, test
import torch.nn as nn
from mak.utils.helper import get_optimizer, log_client_metrics

class FedExClient(BaseClient):
    """
        FedEx client implementation 
    """
    def __init__(self, client_id, model, trainset, valset, config_sim, device):
        super().__init__(client_id, model, trainset, valset, config_sim, device)
        self.hyperparameters = Hyperparameters(self.config_sim['fedex']['hyperparam_config_nr']) #hyperparmeter search space 
        self.hyperparameters.read_from_csv(self.config_sim['fedex']['hyperparam_file'])

    def __repr__(self) -> str:
        return " FedEx client"
    
    def set_parameters_train(self, parameters, config):
        # obtain hyperparams and distribution
        self.distribution = parameters[-1]
        self.hyperparam_config, self.hidx = self._sample_hyperparams()
        
        # remove hyperparameter distribution from parameter list
        parameters = parameters[:-1]
        optimizer = get_optimizer(model=self.model, client_config=config)
        for g in optimizer.param_groups:
            g['lr'] = self.hyperparam_config['learning_rate']
            # g['momentum'] = self.hyperparam_config['momentum']
            # g['weight_decay'] = self.hyperparam_config['weight_decay']

        # self.model.dropout = self.hyperparam_config['dropout']
        
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def set_parameters_evaluate(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters_train(parameters, config)
        batch = config["batch_size"]
        epochs = config["epochs"]
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        valloader = DataLoader(self.valset, batch_size=self.test_batch_size)
        optimizer = get_optimizer(model=self.model, client_config=config)
        
        before_loss, _ = self.test(self.model, valloader, device=self.device)

        self.train(net=self.model, trainloader= trainloader, epochs = epochs, optim = optimizer, device = self.device, config=config)
        after_loss, _ = self.test(self.model, valloader, self.device)
        model_params = self.get_parameters(config=config)
        return model_params, len(trainloader.dataset), {'hidx': self.hidx, 'before': before_loss, 'after': after_loss}

    def evaluate(self, parameters, config):
        self.set_parameters_evaluate(parameters)
        valloader = DataLoader(self.valset, batch_size=self.test_batch_size)
        loss, accuracy = self.test(self.model, valloader, device=self.device)
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

    def _sample_hyperparams(self):
        # obtain new learning rate for this batch
        distribution = torch.distributions.Categorical(torch.FloatTensor(self.distribution))
        hyp_idx = distribution.sample().item()
        print(hyp_idx)
        hyp_config = self.hyperparameters[hyp_idx]
        return hyp_config, hyp_idx


    def train_epoch(self, net, trainloader, criterion, optim, device):
            net.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch in trainloader:
                keys = list(batch.keys())
                x_label, y_label = keys[0], keys[1]
                images, labels = batch[x_label].to(device), batch[y_label].to(device)
                
                optim.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.)
                optim.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(trainloader)
            epoch_accuracy = 100. * correct / total
            return epoch_loss, epoch_accuracy

    def train(self, net, trainloader, optim, epochs, device: str, config: dict):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        samples_per_epoch = len(trainloader.dataset)
        
        for epoch in range(epochs):
            epoch_loss, epoch_accuracy = self.train_epoch(net=net, trainloader=trainloader, criterion=criterion, optim=optim, device=device)
            current_lr = optim.param_groups[0]['lr']

            val_loss, val_samples, val_acc = self.validate(net=net)

            result = {
                "round" : config['round'],
                "epoch": epoch + 1,
                "train_acc": epoch_accuracy,
                "train_loss": epoch_loss,
                "val_acc": val_acc['accuracy'],
                "val_loss": val_loss,
                "lr": current_lr,
                "train_samples": samples_per_epoch, 
                "val_samples": val_samples,
            }
            if self.config_sim['client']['save_train_res']:
                log_client_metrics(out_file_path=self.out_file_path,results=result)
