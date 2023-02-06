import torch

from MedicalDiagnosis.utils import evaluate_model_on_tests
print(torch.Tensor([1,2]).cuda())
# 2 lines of code to change to switch to another dataset
from MedicalDiagnosis.datasets.fed_isic2019 import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS,
    get_nb_max_rounds
)
from MedicalDiagnosis.datasets.fed_isic2019 import FedIsic2019 as FedDataset

# 1st line of code to change to switch to another strategy
from MedicalDiagnosis.strategies.afedsv import FedOptSV as strat

# We loop on all the clients of the distributed dataset and instantiate associated data loaders
train_dataloaders = [
            torch.utils.data.DataLoader(
                FedDataset(center = i, train = True, pooled = False),
                batch_size = BATCH_SIZE,
                shuffle = True,
                num_workers = 0
            )
            for i in range(NUM_CLIENTS)
        ]
full_dataset = FedDataset(train = False, pooled = True)
valid_size = int(0.25 * len(full_dataset))
test_size = len(full_dataset)  - valid_size
print(valid_size, test_size, len(full_dataset))
print(sum([9930,3163,2691,1807,655,351]))
valid_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [valid_size, test_size])
print(len(valid_dataset), len(test_dataset))
test_dataloaders = [
            torch.utils.data.DataLoader(
                test_dataset,
                batch_size = BATCH_SIZE,
                shuffle = False,
                num_workers = 0,
            )
        ]
valid_dataloaders = [
            torch.utils.data.DataLoader(
                valid_dataset,
                batch_size = BATCH_SIZE,
                shuffle = False,
                num_workers = 0,
            )
        ]

lossfunc = BaselineLoss()
m = Baseline()

# Federated Learning loop
# 2nd line of code to change to switch to another strategy (feed the FL strategy the right HPs)
args = {
            "training_dataloaders": train_dataloaders,
            "valid_dataloaders": valid_dataloaders,
            "test_dataloaders": test_dataloaders,
            "model": m,
            "loss": lossfunc,
            "optimizer_class": torch.optim.SGD,
            "learning_rate": 0.01,
            "num_updates": 100,
# This helper function returns the number of rounds necessary to perform approximately as many
# epochs on each local dataset as with the pooled training
            "nrounds": 40,
        }
s = strat(**args)
seeds = [20,21,22,23,24]
for seed in seeds:
    m = s.run(seed)[0]

# Evaluation
# We only instantiate one test set in this particular case: the pooled one

