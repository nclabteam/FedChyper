import torch
from mak.clients.base_client import BaseClient
from mak.hpo.early_stopping import EarlyStopping
from logging import INFO
from flwr.common.logger import log
from mak.utils.helper import get_optimizer, log_client_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FedProxClient(BaseClient):
    """
        Flwr client implementation based on fedprox
        The train loop is changed based on fedprox algorithm 
    """
    def __repr__(self) -> str:
        return " FedProx client"

    # def train(self, net, trainloader, optim, epochs, device: str, config : dict):
    #     """Train the network on the training set for fedprox."""
    #     criterion = torch.nn.CrossEntropyLoss()

    #     global_params = [val.detach().clone() for val in net.parameters()]
    #     net.train()

    #     total_loss = 0
    #     for _ in range(epochs):
    #         epoch_loss = 0
    #         for batch in trainloader:
    #             keys = list(batch.keys())
    #             x_label, y_label = keys[0], keys[1]
    #             images, labels = batch[x_label].to(device), batch[y_label].to(device)
    #             optim.zero_grad()
    #             proximal_term = 0.0
    #             for local_weights, global_weights in zip(net.parameters(), global_params):
    #                 proximal_term += torch.square((local_weights - global_weights).norm(2))
    #             loss = criterion(net(images), labels) + (config['proximal_mu'] / 2) * proximal_term
    #             epoch_loss += loss.item()
    #             loss.backward()
    #             optim.step()
    #         total_loss += (epoch_loss / len(trainloader))
    #     return (total_loss) # total_loss / epochs

    def train_epoch(self, net, trainloader, criterion, optim, global_params,proximal_mu, device):
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
            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += torch.square((local_weights - global_weights).norm(2))

            loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term
            loss.backward()
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
        global_params = [val.detach().clone() for val in net.parameters()]

        proximal_mu = config['proximal_mu']
        samples_per_epoch = len(trainloader.dataset)

        hpo_config = self.config_sim['fedadap']
        if hpo_config['enable']:
            # Common settings
            monitor = hpo_config['common']['monitor']
            mode = hpo_config['common']['mode']
            verbose = hpo_config['common']['verbose']
            
            # Early Stopping
            if hpo_config['early_stopping']['enable']:
                early_stopping = EarlyStopping(
                    patience=hpo_config['early_stopping']['patience_es'],
                    delta=hpo_config['early_stopping']['min_delta'],
                    verbose=verbose
                )

            # Learning Rate Reduction
            if hpo_config['reduce_lr']['enable']:
                lr_scheduler = ReduceLROnPlateau(
                    optimizer=optim,
                    mode=mode,
                    factor=hpo_config['reduce_lr']['factor'],
                    patience=hpo_config['reduce_lr']['patience_lr'],
                    min_lr=hpo_config['reduce_lr']['min_lr'],
                    threshold=hpo_config['reduce_lr']['min_delta'],
                )
        
        for epoch in range(epochs):
            epoch_loss, epoch_accuracy = self.train_epoch(net= net, trainloader=trainloader, criterion=criterion, optim=optim,global_params=global_params,proximal_mu=proximal_mu, device=device)
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

            if hpo_config['enable']:
                if hpo_config['early_stopping']['enable']:
                    early_stopping(val_loss=val_loss,model=net)
                    if early_stopping.early_stop:
                        log(INFO,f"Client ==>:{self.client_id} Early stopping at epoch {epoch+1}")
                        break

                if hpo_config['reduce_lr']['enable']:
                    lr_scheduler.step(val_loss)
                
            