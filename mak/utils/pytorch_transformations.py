from torchvision.transforms import Compose, ToTensor, Normalize
import torch

def apply_transforms_cifar10(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = Compose(
        [
            # Resize(256),
            # CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
        ]
    )
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def apply_transforms_default(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = Compose(
        [
            # Resize(256),
            # CenterCrop(224),
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

def apply_transforms_shakespeare(batch):
    """Apply transforms to the partition from FederatedDataset, returning only x and y as tensors."""
    # Convert x to tensors
    x_tensor = [torch.tensor(data) for data in batch["x"]]
    
    # Convert y to tensors (assuming y needs to be converted as well)
    y_tensor = [torch.tensor(data) for data in batch["y"]]

    # Return a dictionary with only x and y
    return {
        "x": x_tensor,
        "y": y_tensor
    }



def get_transformations(dataset_name):
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        return apply_transforms_cifar10
    elif dataset_name == 'flwrlabs/shakespeare':
        return apply_transforms_shakespeare
    else:
        return  apply_transforms_default
