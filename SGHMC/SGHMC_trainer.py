import torch
import numpy as np

from itertools import islice
import logging

import glob
import copy
import os, sys
from tqdm import tqdm 

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from dataloaders.bcic2a_dataset import get_dataloaders
from SGHMC.sghmc_utils import accuracy, nll
from models.baseline_shallow_model import ShallowModel
from models.layers import safe_log, square
from SGHMC.priors import PriorGaussian
from SGHMC.likelihoods import LikMultinomial
from SGHMC.SGHMC_optimizer import AdaptiveSGHMC
from utils import inf_loop, get_all_data, suppress_stdout, ensure_dir


class SGHMCBayesianNet():
    def __init__(self, model, likelihood, prior, checkpoint_dir, weights_format, device) -> None:
        self.model = model
        
        self.lik_module = likelihood
        self.prior_module = prior
        self.sampled_weights = []
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.weights_format = weights_format
        self.model.to(self.device)
        logging.basicConfig()
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        self.is_trained = False
        self.num_samples = 0
        self.step = 0
        self.num_saved_sets_weights = 0
        self.sampled_weights_dir = os.path.join(self.checkpoint_dir,
                                                "sampled_weights")
        ensure_dir(self.sampled_weights_dir)

    
    @property
    def network_weights(self):
        """Extract current network weight values.
        """
        if self.weights_format == "tuple":
            return tuple(
                np.asarray(parameter.data.clone().detach().cpu().numpy())
                for parameter in self.model.parameters())

        elif self.weights_format == "state_dict":
            return self.model.state_dict()

    @network_weights.setter
    def network_weights(self, weights):
        """Assign new weights to our neural networks parameters.
        """
        if self.weights_format == "tuple":
            for parameter, sample in zip(self.model.parameters(), weights):
                parameter.copy_(torch.from_numpy(sample))

        elif self.weights_format == "state_dict":
            self.model.load_state_dict(weights)
    
    def _initialize_sampler(self, num_datapoints, lr=1e-2, mdecay=0.05,
                            num_burn_in_steps=3000, epsilon=1e-10):
        """Initialize a stochastic gradient MCMC sampler.
        Args:
            num_datapoints: int, the total number of training data points.
            lr: float, learning rate.
            mdecay: float, momemtum decay.
            num_burn_in_steps: int, number of burn-in steps to perform.
                This value is passed to the given `sampler` if it supports
                special burn-in specific behavior like that of Adaptive SGHMC.
            epsilon: float, epsilon for numerical stability. 
        """
        dtype = np.float32
        self.sampler_params = {}

        self.sampler_params['scale_grad'] = dtype(num_datapoints)
        self.sampler_params['lr'] = dtype(lr)
        self.sampler_params['mdecay'] = dtype(mdecay)

        
        self.sampler_params['num_burn_in_steps'] = num_burn_in_steps
        self.sampler_params['epsilon'] = dtype(epsilon)

        self.sampler = AdaptiveSGHMC(self.model.parameters(),
                                        **self.sampler_params)
        
    
    def _neg_log_joint(self, fx_batch, y_batch, num_datapoints):
        """Calculate model's negative log joint density.
            Note that the gradient is computed by: g_prior + N/n sum_i grad_theta_xi.
            Because of that we divide here by N=num of datapoints
            since in the sample we will rescale the gradient by N again.
        Args:
            fx_batch: torch tensor, the predictions.
            y_batch: torch tensor, the corresponding targets.
            num_datapoints: int, the number of data points in the entire
                training set.
        Return:
            The negative log joint density.
        """
        return self.lik_module(fx_batch, y_batch) / y_batch.shape[0] + \
            self.prior_module(self.model) / num_datapoints
    
    def train(self, x_train=None, y_train=None, data_loader=None,
              num_samples=None, keep_every=100, num_burn_in_steps=3000,
              lr=1e-2, batch_size=32, epsilon=1e-10, mdecay=0.05,
              print_every_n_samples=10, continue_training=False):
        '''
        Train a BNN using a given dataset.
        Args:
            x_train: numpy array, input training datapoints.
            y_train: numpy array, input training targets.
            data_loader: instance of DataLoader, the dataloader for training
                data. Notice that we have to choose either numpy arrays or
                dataloader for the input data.
            num_samples: int, number of set of parameters we want to sample.
            keep_every: number of sampling steps (after burn-in) to perform
                before keeping a sample.
            num_burn_in_steps: int, number of burn-in steps to perform.
                This value is passed to the given `sampler` if it supports
                special burn-in specific behavior like that of Adaptive SGHMC.
            lr: float, learning rate.
            batch_size: int, batch size.
            epsilon: float, epsilon for numerical stability.
            mdecay: float, momemtum decay.
            print_every_n_samples: int, defines after how many samples we want
                to print out the statistics of the sampling process.
            continue_training: bool, defines whether we want to continue
                from the last training run.
        '''
        # Setup data loader
        
        num_datapoints = len(data_loader.sampler)
        train_loader = inf_loop(data_loader)
        

        # Estimate the number of update steps
        num_steps = 0 if num_samples is None else (num_samples+1) * keep_every

        # Initialize the sampler
        if not continue_training:
            self.sampled_weights.clear()
            self.model = self.model.float()
            self._initialize_sampler(num_datapoints, lr, mdecay,
                                     num_burn_in_steps, epsilon)
            num_steps += num_burn_in_steps

        # Initialize the batch generator
        batch_generator = islice(enumerate(train_loader), num_steps)

        # Start sampling
        self.model.train()
        for step, (x_batch, y_batch) in batch_generator:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            
            # Forward pass
            fx_batch = self.model(x_batch)

            self.sampler.zero_grad()
            # Calculate the negative log joint density
            loss = self._neg_log_joint(fx_batch, y_batch, num_datapoints)

            # Estimate the gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.)

            # Update parameters
            self.sampler.step()
            self.step += 1

            # Save the sampled weight
            if (step > num_burn_in_steps) and \
                    ((step - num_burn_in_steps) % keep_every == 0):
                self.sampled_weights.append(copy.deepcopy(self.network_weights))
                self.num_samples += 1

                # Print evaluation on training data
                if self.num_samples % print_every_n_samples == 0:
                    self.model.eval()
                    if (x_train is not None) and (y_train is not None):
                        self._print_evaluations(x_train, y_train, True)
                    else:
                        self._print_evaluations(x_batch, y_batch, True)
                    self.model.train()
    
    def evaluate(self, test_data_loader, return_individual_predictions=True,
                 all_sampled_weights=False):
        """Evaluate the sampled weights on a given test dataset.
        Args:
            test_data_loader: instance of data loader, the data loader for
                test data.
            return_individual_predictions: bool, if True also the predictions
                of the individual models are returned.
            all_sampled_weights: bool, if True loadd all sampled weights from
                file to make predictions; otherwise, only use the current
                sampled weights in the lists `self.sampled_weights`.
        Returns:
            torch tensor, the predicted mean.
        """
        self.model.eval()

        def network_predict(test_data_loader_, weights):
            predictions = []
            with torch.no_grad():
                self.network_weights = weights

                for x_batch, y_batch in test_data_loader_:
                    x_batch = x_batch.to(self.device)
                    predictions.append(self.model(x_batch).float())

                return torch.cat(predictions, dim=0)

        # Make predictions
        network_outputs = []

        if all_sampled_weights:
            sampled_weights_loader = self._load_all_sampled_weights()
            for weights in tqdm(sampled_weights_loader):
            # for weights in sampled_weights_loader:
                network_outputs.append(network_predict(test_data_loader,
                                                       weights=weights))
        else:
            for weights in self.sampled_weights:
                network_outputs.append(network_predict(test_data_loader,
                                                       weights=weights))

        predictions = torch.stack(network_outputs, dim=0)

        # Estimate the predictive mean
        mean_predictions = torch.mean(predictions, axis=0)
        mean_predictions = mean_predictions.to(self.device)

        if return_individual_predictions:
            return mean_predictions, predictions

        return mean_predictions

    def predict(self, x_test, all_sampled_weights=True):
        """Evaluate the sampled weights on a given test dataset.
        Args:
            x_test: torch tensor, the test datapoint.
            return_individual_predictions: bool, if True also the predictions
                of the individual models are returned.
        Returns:
            torch tensor, the predicted mean.
        """
        self.model.eval()

        def network_predict(x_test_, weights, device):
            with torch.no_grad():
                self.network_weights = weights
                return self.model(x_test_.to(device))

        # Make predictions
        network_outputs = []

        if all_sampled_weights:
            sampled_weights_loader = self._load_all_sampled_weights()
            for weights in sampled_weights_loader:
                network_outputs.append(network_predict(
                    x_test, weights, self.device))
        else:
            for weights in self.sampled_weights:
                network_outputs.append(network_predict(
                    x_test, weights, self.device))

        predictions = torch.stack(network_outputs, dim=0)

        # Estimate the predictive mean
        mean_predictions = torch.mean(predictions, axis=0)
        mean_predictions = mean_predictions.to(self.device)

        return mean_predictions
    
    def train_and_evaluate(self, data_loader, valid_data_loader,
                           num_samples=1000, keep_every=100, lr=1e-2,
                           mdecay=0.05, batch_size=20, num_burn_in_steps=3000,
                           validate_every_n_samples=10, print_every_n_samples=5,
                           epsilon=1e-10, continue_training=False):
        """
        Train and validates the neural network
        Args:
            data_loader: instance of DataLoader, the dataloader for training
                data.
            valid_data_loader: instance of DataLoader, the data loader for
                validation data.
            num_samples: int, number of set of parameters we want to sample.
            keep_every: number of sampling steps (after burn-in) to perform
                before keeping a sample.
            num_burn_in_steps: int, number of burn-in steps to perform.
                This value is passed to the given `sampler` if it supports
                special burn-in specific behavior like that of Adaptive SGHMC.
            lr: float, learning rate.
            batch_size: int, batch size.
            epsilon: float, epsilon for numerical stability.
            mdecay: float, momemtum decay.
            continue_training: bool, defines whether we want to continue
                from the last training run.
        """
        if not continue_training:
            # Burn-in steps
            self.logger.info("Burn-in steps")
            self.train(data_loader=data_loader, lr=lr, epsilon=epsilon,
                       mdecay=mdecay, num_burn_in_steps=num_burn_in_steps)

        _, valid_targets = get_all_data(valid_data_loader)
        valid_targets = valid_targets.to(self.device)

        best_nll = None
        preds = [] # List containing prediction
        self.logger.info("Start sampling")
        for i in range(num_samples // validate_every_n_samples):
            self.train(data_loader=data_loader, num_burn_in_steps=0,
                       num_samples=validate_every_n_samples,
                       batch_size=batch_size,
                       lr=lr, epsilon=epsilon, mdecay=mdecay,
                       keep_every=keep_every, continue_training=True,
                       print_every_n_samples=print_every_n_samples)
            self._save_sampled_weights()

            # Make predictions
            _, new_preds = self.evaluate(valid_data_loader)
            preds.append(new_preds)

            # Evaluate the sampled weights on validation data
            mean_preds = torch.cat(preds, dim=0).mean(axis=0)
            nll_ = self.lik_module(mean_preds, valid_targets) / valid_targets.shape[0]
            accuracy_ = accuracy(mean_preds, valid_targets)

            # Save the best checkpoint
            if (best_nll is None) or (nll_ <= best_nll):
                best_nll = nll_
                self._save_checkpoint(mode="best")

            self.logger.info("Validation: NLL = {:.5f} Acc = {:.4f}".format(
                nll_, accuracy_))
            self._save_checkpoint(mode="last")

            # Clear the cached weights
            self.sampled_weights.clear()

        self.logger.info("Finish")

    
    def _save_sampled_weights(self):
        """Save a set of sampled weights to file.
        Args:
            sampled_weights: a state_dict containing the model's parameters.
        """
        file_path = os.path.join(self.sampled_weights_dir,
                                 "sampled_weights_{0:07d}".format(
                                     self.num_saved_sets_weights))
        torch.save({"sampled_weights": self.sampled_weights}, file_path)
        self.num_saved_sets_weights += 1
        
    def _print_evaluations(self, x, y, train=True):
        """Evaluate the sampled weights on training/validation data and
            during the training log the results.
        Args:
            x: numpy array, shape [batch_size, num_features], the input data.
            y: numpy array, shape [batch_size, 1], the corresponding targets.
            train: bool, indicate whether we're evaluating on the training data.
        """
        preds = self.predict(x, all_sampled_weights=(not train))
        acc_ = accuracy(preds, y)
        # nll_ = nll(preds, y)
        nll_ = self.lik_module(preds,y)/y.shape[0]

        if train:
            self.logger.info("Samples # {:5d} : NLL = {:.5f} "
                "Acc = {:.4f} ".format(self.num_samples, nll_, acc_))
        else:
            self.logger.info("Validation: NLL = {:.5f} Acc = {:.4f}".format(
                nll_, acc_))
        
    def _save_checkpoint(self, mode="best"):
        """Save sampled weights, sampler state into a single checkpoint file.
        Args:
            mode: str, the type of checkpoint to be saved. Possible values
                `last`, `best`.
        """
        if mode == "best":
            file_name = "checkpoint_best.pth"
        elif mode == "last":
            file_name = "checkpoint_last.pth"
        else:
            file_name = "checkpoint_step_{}.pth".format(self.step)

        file_path = os.path.join(self.checkpoint_dir, file_name)

        torch.save({
            "step": self.step,
            "num_samples": self.num_samples,
            "num_saved_sets_weights": self.num_saved_sets_weights,
            "sampler_params": self.sampler_params,
            "model_state_dict": self.model.state_dict(),
            "sampler_state_dict": self.sampler.state_dict(),
        }, file_path)
        
    def _load_all_sampled_weights(self):
        """Load all the sampled weights from files.
        Returns: a generator for loading sampled weights.
        """
        def load_weights(file_path):
            checkpoint = torch.load(file_path)
            sampled_weights = checkpoint["sampled_weights"]

            return sampled_weights

        def sampled_weights_loader(sampled_weights_dir):
            file_paths = glob.glob(os.path.join(sampled_weights_dir,
                                                "sampled_weights*"))
            for file_path in file_paths:
                for weights in load_weights(file_path):
                    yield weights

                self.network_weights.clear()
                if "cuda" in str(self.device):
                    torch.cuda.empty_cache()

        return sampled_weights_loader(self.sampled_weights_dir)

if __name__ == '__main__':
    device = 'cuda:0'
    
    # subject = 8
    for subject in [3,4,5,6,9]:
        checkpoint_dir = os.path.join(SCRIPT_DIR, PACKAGE_PARENT, 'checkpoints/bcic_shallow_SGHMC/subj{}'.format(subject))
        with suppress_stdout():
            train_dataloader, test_dataloader, _, n_chans, input_time_length = get_dataloaders(subjects=[subject], batch_size=64,
                                                                                               events=[0,1], permute=(0,2,1), insert_rgb_dim=True)
        model = ShallowModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=2, conv_nonlin=square, pool_nonlin=safe_log)

        prior = PriorGaussian(sw2=1) #
        likelihood = LikMultinomial()
        net = SGHMCBayesianNet(model, likelihood, prior, checkpoint_dir, weights_format='state_dict', device=device)
        net.train_and_evaluate(data_loader=train_dataloader, valid_data_loader=test_dataloader)