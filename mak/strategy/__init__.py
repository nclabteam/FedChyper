from flwr.server.strategy import (
    FedProx,
    FedAvgM,
    FedOpt,
    FedAdam,
    FedMedian,
)
from mak.strategy.fedlaw_strategy import FedLaw
from mak.strategy.fedex_strategy import FedEx
from mak.strategy.fedams_strategy import FedAms
from mak.strategy.fedlbw_strategy import FedLbw
from mak.strategy.fedavg_straggler_strategy import FedAvgS as FedAvg