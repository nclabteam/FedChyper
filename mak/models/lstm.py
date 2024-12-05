from mak.models.base_model import Model
from torch import nn, Tensor
from torch import load

class LSTMModel(Model):
    """StackedLSTM architecture.

    As described in Fei Chen 2018 paper :

    [FedMeta: Federated Meta-Learning with Fast Convergence and Efficient Communication]
    (https://arxiv.org/abs/1802.07876)
    """

    def __init__(self, num_classes: int, weights = None, *args, **kwargs) -> None:
        super().__init__(num_classes, *args, **kwargs)

        self.embedding = nn.Embedding(self.input_shape, 8)
        self.lstm = nn.LSTM(8, 256, num_layers=2, dropout=0.5, batch_first=True)
        self.fully_ = nn.Linear(256, self.num_classes)

    def forward(self, text):
        """Forward pass of the StackedLSTM.

        Parameters
        ----------
        text : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        embedded = self.embedding(text)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embedded)
        final_output = self.fully_(lstm_out[:, -1, :])
        return final_output