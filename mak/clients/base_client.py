import torch
from torch.utils.data import DataLoader
import flwr as fl
from mak.utils.helper import get_optimizer, log_client_metrics, gen_dir_outfile_client
from mak.utils.general import set_params, test
from mak.hpo.early_stopping import EarlyStopping
from logging import INFO
from flwr.common.logger import log
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BaseClient(fl.client.NumPyClient):
    """flwr base client implementaion """
    def __init__(self, client_id, model, trainset, valset, config_sim, device, straggler_schedule):
        self.client_id = client_id
        self.trainset = trainset
        self.valset = valset
        self.model = model
        self.device = device
        self.config_sim = config_sim
        self.train_batch_size = config_sim['client']['batch_size']
        self.test_batch_size = config_sim['client']['test_batch_size']
        self.model.to(self.device)
        self.straggler_schedule = straggler_schedule

        if self.config_sim['client']['save_train_res']:
            self.out_file_path =  gen_dir_outfile_client(cid=self.client_id,config=self.config_sim)
        else:
            self.out_file_path = None

    def __repr__(self) -> str:
        return " Flwr base client"

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        set_params(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
            
        batch, epochs, learning_rate = config["batch_size"], config["epochs"], config["lr"]
        
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        optimizer = get_optimizer(model=self.model, client_config=config)
        self.train(net=self.model, trainloader=trainloader, optim=optimizer, epochs=epochs, device=self.device, config = config)

        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        valloader = DataLoader(self.valset, batch_size=self.test_batch_size)
        loss, accuracy = self.test(self.model, valloader, device=self.device)
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
    
    def validate(self, net):
        valloader = DataLoader(self.valset, batch_size=self.test_batch_size)
        loss, accuracy = self.test(net, valloader, device=self.device)
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
    
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
                    threshold=hpo_config['reduce_lr']['min_delta']
                )
        
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

            if hpo_config['enable']:
                if hpo_config['reduce_lr']['enable']:
                    lr_scheduler.step(val_loss)

                if hpo_config['early_stopping']['enable']:
                    early_stopping(val_loss=val_loss,model=net)
                    if early_stopping.early_stop:
                        log(INFO,f"Client ==>:{self.client_id} Early stopping at epoch {epoch+1}")
                        break

                
                
            
            
            
        
    
    # def train(self, net, trainloader, optim, epochs, device: str, config : dict):
    #     """Train the network on the training set."""
    #     criterion = torch.nn.CrossEntropyLoss()
    #     net.train()

    #     running_loss = 0.0
    #     correct = 0
    #     total = 0

    #     es = EarlyStopping(patience=3,delta=0.01)

    #     for e in range(epochs):
    #         for batch in trainloader:
    #             keys = list(batch.keys())
    #             x_label, y_label = keys[0], keys[1]
    #             images, labels = batch[x_label].to(device), batch[y_label].to(device)
    #             optim.zero_grad()
    #             outputs = net(images)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optim.step()

    #             running_loss += loss.item()
    #             _, predicted = outputs.max(1)
    #             total += labels.size(0)
    #             correct += predicted.eq(labels).sum().item()

    #         es(val_loss=loss,model=net)
    #         if es.early_stop:
    #             print("+++++++++++ Early stopping at epoch ",e)
    #             break
    

    def test(self, net, testloader, device: str):
       return test(net=net,testloader=testloader,device=device)



