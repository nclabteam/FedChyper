from torch.utils.data import DataLoader
from mak.utils.helper import get_optimizer
from mak.clients.base_client import BaseClient
import numpy as np

class FedAvgClient(BaseClient):
    """
        Simple flwr client implementation using basic fedavg approach
    """
    def __repr__(self) -> str:
        return " FedAvg client"

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        batch, epochs, learning_rate, use_straggler = config["batch_size"], config["epochs"], config["lr"], config['use_straggler']
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        if use_straggler:
            if (
                self.straggler_schedule[int(config["round"]) - 1]
                and epochs > 1
            ):
                num_epochs = np.random.randint(1, epochs)

                if config["drop_straggler"]:
                    # return without doing any training.
                    # The flag in the metric will be used to tell the strategy
                    # to discard the model upon aggregation
                    return (
                        self.get_parameters({}),
                        len(trainloader),
                        {"is_straggler": True},
                    )
            else:
                num_epochs = epochs
        else:
            num_epochs = epochs
        optimizer = get_optimizer(model=self.model, client_config=config)
        self.train(net=self.model, trainloader=trainloader, optim=optimizer, epochs=num_epochs, device=self.device,config = config)

        return self.get_parameters({}), len(trainloader.dataset), {"is_straggler": False}