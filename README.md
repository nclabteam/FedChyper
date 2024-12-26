## FedChyper

### Note: This code is written and tested on Ubuntu and can work easily on any linux based distribution. For windows users some steps needs to changed.

## Initial Setup

1. Clone this repository using command
```bash
 git clone https://github.com/nclabteam/FedChyper.git
```
2. After clone cd into cloned directory and open terminal.

3. Ensure pip is installed if not install using
```bash
 sudo apt install python3-pip
```

4. We will be using miniconda to create a virtual environment, download miniconda as
If You already have conda installed skip step 4 and 5.

```bash
 curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
```
5. Then install miniconda using below command
```bash
  bash Miniconda3-latest-Linux-x86_64.sh
```

6. Create a new virtual environment using conda
```bash
 conda env create -f environment.yaml
```
It will create a virtual env named `venv-flwr` based on `environment.yaml` file

7. For using virtual environment we need to activate the environment first.
```bash
 conda deactivate
 conda activate venv-flwr
```

## Running the Experiment

This repository uses `config.yaml` for confugration and we can change the confugration as per our need in `config.yaml` file. Change the dataset, data distribution, ml model and other things in `config.yaml` and run or replace the contents `./config.yaml` with any example config file from `./example_configs`. 

```bash
python run_simulation.py
```
This script will read the confugration from `config.yaml` file and starts the simulation.

The outputs will be saved in `out` directory.

## Sample Config files
The different confugrations that we used in evaluations are given as saperate `config.yaml` files in `./example_configs` directory

### Image Classification Using Cifar-10 dataset
CIFAR-10 dataset is downloaded automatically from Flwr Datasets.

* [`FedAvg -> CIFAR-10 with non-IID Data (alpha = 0.1)`](/example_configs/ex_fedavg_config_cifar_niid_0_1.yaml)

* [`FedAvg + FedChyper -> CIFAR-10 with non-IID Data (alpha = 0.1)`](/example_configs/ex_fedadap_config_cifar_niid_0_5.yaml)


### Next Character Prediction on Shakespeare Dataset

The shakespeare dataset can be downloaded and preprocessed by running 
```bash
 python ./preprocessing/preprocess_shakespeare.py 
```

* [`FedAvg -> Next Character Prediction Shakespeare`](/example_configs/ex_fedavg_config_shakespeare.yaml)

* [`FedAvg + FedChyper -> Next Character Prediction Shakespeare`](/example_configs/ex_fedadap_config_shakespeare.yaml)



These example config files will evaluate FedAvg and FedAvg + FedChyper on the settings. 
If you want to test other algorithms change the strategy accordingly in the config file.


## Graph Plotting
We can use `graphs_fedchyper.ipynb` to generate the graphs, but for that we need to consolidate the data please check `graphs_chyper.ipynb` for details.

## Description about  [`config.yaml`](/config.yaml) file
The `config.yaml` file is a configuration file for this framework that trains a Federated Learning model.

The configuration file is divided into three sections: `common`, `server`, and `client`.

### Common Section
The `common` section contains the common configurations used in this framework. 

- `data_type` : This field specifies the data distribution type used in the training process. Currently supported data distributions are [`iid`,`dirichlet_niid`] .
- `dataset` : This field specifies the dataset used in the training process. Currently supported data distributions are [`mnist`, `cifar10`, `fashion_mnist`, `sasha/dog-food`, `zh-plus/tiny-imagenet`,`flwrlabs/shakespeare`]. 
- `dirichlet_alpha` : This field is used when `data_type` is set to `dirichlet_niid`. It specifies the Dirichlet concentration parameter.
- `target_acc` : This field specifies the target accuracy that the model needs to achieve. It can take any value greater than `0`.
- `model` : This field specifies the model architecture used in the training process. Currently Implemented models are [ `Net`, `CifarNet`, `SimpleCNN`, `KerasExpCNN`, `MNISTCNN`, `SimpleDNN`, `FMCNNModel`,`FedAVGCNN`,`Resnet18`, `Resnet34`,`ResNet18Pretrained`, `ResNet34Pretrained`,`ResNet18Small`, `ResNet20Small`,`MobileNetV2`,`EfficientNetB0`,`LSTMModel`]. 
- `optimizer` : This field specifies the optimizer used in the training process. It could be either `sgd` or `adam`.
- `seed` : This field fixes the seed for reproducibility

### Server Section
The `server` section contains the configurations for the server that coordinates the Federated Learning process.

- `max_rounds` : This field specifies the maximum number of rounds for the training process.
- `address` : This field specifies the IP address of the server.
- `fraction_fit` : This field specifies the fraction of participating clients used for training in each round.
- `min_fit_clients` : This field specifies the minimum number of participating clients required for training in each round.
- `num_clients` : Total number of clients participating in training.
- `fraction_evaluate` : This field specifies the fraction of participating clients used for evaluation in each round.
- `min_avalaible_clients` : This field specifies the minimum number of clients that should be available for the training process.
- `strategy` : This field specifies the strategy used for Federated Learning. Currently supported strategies are [`FedLaw`, `FedProx`, `FedAvgM`, `FedOpt`, `FedAdam`, `FedMedian`, `FedAvg`,] 

### Client Section
The `client` section contains the configurations for the clients participating in the Federated Learning process.

- `epochs` : This field specifies the number of epochs for each client's training process.
- `batch_size` : This field specifies the batch size for each client's training process.
- `lr` : This field specifies the learning rate for each client's training process.
- `save_train_res` : This field specifies whether to save the training results. It could be either `true` or `false`.
If `save_train_res` is set to `true`, all the output data like accuracy, loss, time of each round would be saved in the `out` directory.
- `total_cpus` : No. of CPU cores that are assigned for all simulation
- `total_gpus` :  No. of GPU's assigned for whole simulation
- `gpu` : True or False, Use GPU for training or not. Default to False
- `num_cpus` : No. of CPU cores that are assigned for each actor Default to 1
- `num_gpus` : Fraction of GPU assigned to each actor. (num_cpus and num_gpus can only used in simulation mode if `simulation` is set to `True`) For more details on this please refer to https://flower.dev/docs/framework/how-to-run-simulations.html and https://docs.ray.io/en/latest/ray-core/scheduling/resources.html

### Shakespeare
The `shakespeare` section is only needed if you need to evaluvate shakespeare dataset for next character prediction task.

- `file_path` : path of processed shakespare file

### FedChyper
The `FedChyper` sections defines the parameters that are used by the FedChyper algorithm.

- `enable` : This field enables or disables the FedChyper. When set to True, it augments the selected strategy with FedChyper.
- `early_stopping` : block for config of early stopping part.
  - `enable` : This field enables or disables early stopping mechanism. When True, it prevents model overfitting by stopping training when performance plateaus.
  - `patience_es` : This field specifies the number of epochs to wait for improvement before stopping the training. In this case, it is set to 6 epochs.
  - `min_delta` : This field defines the minimum change in the monitored metric to qualify as an improvement. It is set to 0.01 to distinguish between significant and negligible changes.
- `reduce_lr` : block for config of dynamic learning rate of FedChyper.
  - `enable` : This field enables or disables dynamic learning rate reduction. When True, it helps the model converge by gradually reducing the learning rate.
  - `patience_lr` : This field specifies the number of epochs to wait before reducing the learning rate. It is set to 3 epochs, allowing quicker adaptation.
  - `factor` : This field sets the multiplication factor for learning rate reduction. Set to 0.9.
  - `min_lr` : This field sets the minimum limit for learning rate reduction. Set to 0.0001, it prevents the learning rate from becoming too small.



