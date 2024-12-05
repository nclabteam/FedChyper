import torch
import flwr as fl
from typing import Tuple, List
from flwr.common import Metrics
from collections import OrderedDict

# borrowed from Pytorch quickstart example
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            keys = list(data.keys())
            x_label, y_label = keys[0], keys[1]
            images, labels = data[x_label].to(device), data[y_label].to(device)
            outputs = net(images)
            # loss += criterion(outputs, labels).item()
            loss += criterion(outputs, labels).item() * labels.size(0)  # Scale by batch size
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    loss = loss / len(testloader.dataset)  # Normalize by total number of samples
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def filter_straggler_results(results):
    """Discard all the models sent by the clients that were stragglers."""
    # Record which client was a straggler in this round
    stragglers_mask = [res.metrics["is_straggler"] for _, res in results]

    # keep those results that are not from stragglers
    results = [res for i, res in enumerate(results) if not stragglers_mask[i]]
    return results