import time
from typing import List

import torch
import copy
from tqdm import tqdm
import json
from Shapley import Shapley
from noise import GradientNoise
from MedicalDiagnosis.strategies.utils import DataLoaderWithMemory, _Model
from MedicalDiagnosis.utils import evaluate_model_on_tests
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
class FedOptSV:
    """Federated Averaging Strategy class.

    The Federated Averaging strategy is the most simple centralized FL strategy.
    Each client first trains his version of a global model locally on its data,
    the states of the model of each client are then weighted-averaged and returned
    to each client for further training.

    References
    ----------
    - https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    training_dataloaders : List
        The list of training dataloaders from multiple training centers.
    model : torch.nn.Module
        An initialized torch model.
    loss : torch.nn.modules.loss._Loss
        The loss to minimize between the predictions of the model and the
        ground truth.
    optimizer_class : torch.optim.Optimizer
        The class of the torch model optimizer to use at each step.
    learning_rate : float
        The learning rate to be given to the optimizer_class.
    num_updates : int
        The number of updates to do on each client at each round.
    nrounds : int
        The number of communication rounds to do.
    dp_target_epsilon: float
        The target epsilon for (epsilon, delta)-differential
        private guarantee. Defaults to None.
    dp_target_delta: float
        The target delta for (epsilon, delta)-differential private
        guarantee. Defaults to None.
    dp_max_grad_norm: float
        The maximum L2 norm of per-sample gradients; used to enforce
        differential privacy. Defaults to None.
    log: bool, optional
        Whether or not to store logs in tensorboard. Defaults to False.
    log_period: int, optional
        If log is True then log the loss every log_period batch updates.
        Defauts to 100.
    bits_counting_function : Union[callable, None], optional
        A function making sure exchanges respect the rules, this function
        can be obtained by decorating check_exchange_compliance in
        MedicalDiagnosis.utils. Should have the signature List[Tensor] -> int.
        Defaults to None.
    logdir: str, optional
        Where logs are stored. Defaults to ./runs.
    log_basename: str, optional
        The basename of the created log_file. Defaults to fed_avg.
    """

    def __init__(
        self,
        training_dataloaders: List,
        test_dataloaders: List,
        valid_dataloaders: List,
        model: torch.nn.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer_class: torch.optim.Optimizer,
        learning_rate: float,
        num_updates: int,
        nrounds: int,
        dp_target_epsilon: float = None,
        dp_target_delta: float = None,
        dp_max_grad_norm: float = None,
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
        logdir: str = "./runs",
        log_basename: str = "fed_avg",
        seed=None,
    ):
        """
        Cf class docstring
        """
        self._seed = seed if seed is not None else int(time.time())
        self.test_dataloaders = test_dataloaders
        self.valid_dataloaders = valid_dataloaders
        self.training_dataloaders_with_memory = [
            DataLoaderWithMemory(e) for e in training_dataloaders
        ]
        self.training_sizes = [len(e) for e in self.training_dataloaders_with_memory]
        self.total_number_of_samples = sum(self.training_sizes)

        self.dp_target_epsilon = dp_target_epsilon
        self.dp_target_delta = dp_target_delta
        self.dp_max_grad_norm = dp_max_grad_norm

        self.log = log
        self.log_period = log_period
        self.log_basename = log_basename
        self.logdir = logdir

        self.models_list = [
            _Model(
                model=model,
                optimizer_class=optimizer_class,
                lr=learning_rate,
                train_dl=_train_dl,
                dp_target_epsilon=self.dp_target_epsilon,
                dp_target_delta=self.dp_target_delta,
                dp_max_grad_norm=self.dp_max_grad_norm,
                loss=loss,
                nrounds=nrounds,
                log=self.log,
                client_id=i,
                log_period=self.log_period,
                log_basename=self.log_basename,
                logdir=self.logdir,
                seed=self._seed,
            )
            for i, _train_dl in enumerate(training_dataloaders)
        ]
        self.nrounds = nrounds
        self.num_updates = num_updates
        

        self.num_clients = len(self.training_sizes)
        self.bits_counting_function = bits_counting_function

    def _local_optimization(self, _model: _Model, dataloader_with_memory):
        """Carry out the local optimization step.

        Parameters
        ----------
        _model: _Model
            The model on the local device used by the optimization step.
        dataloader_with_memory : dataloaderwithmemory
            A dataloader that can be called infinitely using its get_samples()
            method.
        """
        _model._local_train(dataloader_with_memory, self.num_updates)

    def perform_round(self):
        """Does a single federated averaging round. The following steps will be
        performed:

        - each model will be trained locally for num_updates batches.
        - the parameter updates will be collected and averaged. Averages will be
          weighted by the number of samples in each client
        - the averaged updates willl be used to update the local model
        """
        local_states = list()
        noise = 5
        noiselevel = 0.1
        org = copy.deepcopy(self.models_list[0].model.state_dict())
        for _model, dataloader_with_memory, size in zip(
            self.models_list, self.training_dataloaders_with_memory, self.training_sizes
        ):
            # Local Optimization
            _local_previous_state = _model._get_current_params()
            self._local_optimization(_model, dataloader_with_memory)
            _local_next_state = _model._get_current_params()
            local_states.append(copy.deepcopy(_model.model.state_dict()))
            del _local_next_state

            # Reset local model
            for p_new, p_old in zip(_model.model.parameters(), _local_previous_state):
                p_new.data = torch.from_numpy(p_old).to(p_new.device)
                self.device = p_new.device
            del _local_previous_state

            if self.bits_counting_function is not None:
                self.bits_counting_function(updates)
        local_states = GradientNoise (local_states, noise, noiselevel, self.device, self.seed)
        # Aggregation step
        aggregated_delta_weights = [
            None for _ in range(len(local_states[0]))
        ]
        # for idx in range(6):
        #     print(len(local_weights[idx]),type(local_weights[idx]))
        Fed_sv = Shapley(local_states, self.models_list[0].model, self.valid_dataloaders, 0)
        shapley = Fed_sv.eval_ccshap_stratified(5)

        min_shapley = min(shapley)
        max_shapley = max(shapley)
        gamma = 0.3
        for i in range(len(shapley)):
            self.global_shapley[i] = (1 - gamma) * self.global_shapley[i] + gamma*(shapley[i]-min_shapley)/(max_shapley-min_shapley)

        tmpshapley = [self.global_shapley[i] / float(sum(self.global_shapley)) for i in range(len(self.global_shapley))]
        print(tmpshapley)


        w_avg = copy.deepcopy(org)
        for key in w_avg.keys():
            for i in range(0, len(local_states)):
                w_avg[key] = w_avg[key] + (local_states[i][key]-org[key])*tmpshapley[i]
        
        # Update models
        for _model in self.models_list:
            _model.model.load_state_dict(w_avg)
        
        dict_cindex = evaluate_model_on_tests(self.models_list[0].model, self.test_dataloaders, metric)
        print(dict_cindex)
        path = "fedoptsv{}_{}_{}.txt".format(self.seed, noise, noiselevel)
        f = open(path, "a+")
        print(dict_cindex)
        js = json.dumps(dict_cindex)
        f.write(js)
        f.write('\r\n')
        f.close()

    def run(self):
        """This method performs self.nrounds rounds of averaging
        and returns the list of models.
        """
        l = [9930,3163,2691,1807,655,351]
        self.global_shapley = []
        for ele in l:
            self.global_shapley.append(ele / sum(l))
        seeds = range(20,26)
        for seed in seeds: 
            self.seed = seed
            for _ in tqdm(range(self.nrounds)):
                self.perform_round()
        return [m.model for m in self.models_list]
