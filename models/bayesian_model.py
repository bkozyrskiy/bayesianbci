from click import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Expression, logmeanexp
from .layers import BBBConv2d, BBBLinear, ModuleWrapper
import metrics
import numpy as np


class LLBayesianShallowModel(ModuleWrapper):
    def __init__(self, in_chans, input_time_length, n_classes, n_filters_time=40, 
                 filter_time_length=25, n_filters_spat=40, pool_time_length=75, pool_time_stride=15, conv_nonlin=None, pool_nonlin=None, priors=None) -> None:
        super(LLBayesianShallowModel, self).__init__()
        self.n_classes = n_classes
        
        self.model = nn.Sequential()
        # self.model.add_module("conv_time", BBBConv2d(1, n_filters_time, (filter_time_length , 1), stride=1, bias=True, priors=priors))
        # self.model.add_module("conv_spat", BBBConv2d(n_filters_time, n_filters_spat, (1, in_chans), stride=1, bias=True, priors=priors))
        self.model.add_module("conv_time", nn.Conv2d(1, n_filters_time, (filter_time_length , 1), stride=1, bias=True))
        self.model.add_module("conv_spat", nn.Conv2d(n_filters_time, n_filters_spat, (1, in_chans), stride=1, bias=True))
        self.model.add_module("conv_nonlin", Expression(conv_nonlin))
        self.model.add_module('pool', nn.AvgPool2d(kernel_size=(pool_time_length, 1),
                                                                stride=(pool_time_stride, 1)))
        self.model.add_module("pool_nonlin", Expression(pool_nonlin))
        with torch.no_grad():
            out_feat_dim = self.model(torch.ones(1, 1, input_time_length, in_chans))
        self.final_conv_length = out_feat_dim.shape[2]
        self.model.add_module(
            "conv_classifier",
            BBBConv2d(
                n_filters_spat,
                n_classes,
                (self.final_conv_length, 1),
                bias=False, priors=priors
            ),
        )
        self.model.add_module('flatten',nn.Flatten())

class BayesianShallowModel(ModuleWrapper):
    def __init__(self, in_chans, input_time_length, n_classes, n_filters_time=40, 
                 filter_time_length=25, n_filters_spat=10, pool_time_length=75, pool_time_stride=15, conv_nonlin=None, pool_nonlin=None, priors=None) -> None:
        super(BayesianShallowModel, self).__init__()
        self.n_classes = n_classes
        
        self.model = nn.Sequential()
        # self.model.add_module("conv_time", BBBConv2d(1, n_filters_time, (filter_time_length , 1), stride=1, bias=True, priors=priors))
        # self.model.add_module("conv_spat", BBBConv2d(n_filters_time, n_filters_spat, (1, in_chans), stride=1, bias=True, priors=priors))
        self.model.add_module("conv_time", nn.Conv2d(1, n_filters_time, (filter_time_length , 1), stride=1, bias=True))
        self.model.add_module("conv_spat", BBBConv2d(n_filters_time, n_filters_spat, (1, in_chans), stride=1, bias=True, priors=priors))
        self.model.add_module("conv_nonlin", Expression(conv_nonlin))
        self.model.add_module('pool', nn.AvgPool2d(kernel_size=(pool_time_length, 1),
                                                                stride=(pool_time_stride, 1)))
        self.model.add_module("pool_nonlin", Expression(pool_nonlin))
        with torch.no_grad():
            out_feat_dim = self.model(torch.ones(1, 1, input_time_length, in_chans))
        self.final_conv_length = out_feat_dim.shape[2]
        self.model.add_module(
            "conv_classifier",
            BBBConv2d(
                n_filters_spat,
                n_classes,
                (self.final_conv_length, 1),
                bias=False, priors=priors
            ),
        )
        self.model.add_module('flatten',nn.Flatten())
  



  
class Trainer():
    def __init__(self, model, optim, train_dataloader, test_dataloader, beta_type, nmc_train, nmc_test, device, verbose=False) -> None:
        self.model = model
        self.optim = optim
        self.device = device
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.verbose = verbose
        self.nmc_train = nmc_train
        self.nmc_test = nmc_test
        self.beta_type = beta_type
    
    def _get_predictions_std(self,):
        self.model.eval()
        self.model.sampling(True)
        for test_data in iter(self.test_dataloader):
            x_test, y_test = test_data[0], test_data[1]
            y_test = torch.eye(self.model.n_classes)[y_test]
            x_test, y_test = self.prepare_dim(x_test).to(self.device), y_test.to(self.device)
            logits = torch.zeros((self.nmc_train, x_test.shape[0], self.model.n_classes),
                             device=self.device)
            for s in range(self.nmc_test):
                logits[s,:,:] = self.model.forward(x_test)
            
                
    # def _get_predictions_entropy(self,):
    
    
    def test(self):
        self.model.eval()
        self.model.sample(False)
        with torch.no_grad():
            test_nll, test_error = 0,0 
            for test_data in self.test_dataloader:
                x_test, y_test = test_data[0], test_data[1]
                y_test = torch.eye(self.model.n_classes)[y_test]
                x_test, y_test = self.prepare_dim(x_test).to(self.device), y_test.to(self.device)
                net_out, _ = self.model.forward(x_test)
                logits_test = net_out
                test_nll += -metrics.softmax_log_likelihood(logits_test, y_test).sum()
                test_error += self.compute_error(logits_test, y_test)
            test_nll /= len(self.test_dataloader)
            test_error /= len(self.test_dataloader)
        return test_nll, test_error   
    
    def compute_error(self,logits, y_true): 
        return (torch.argmax(logits, dim=1) != torch.argmax(y_true, dim=1)).float().mean()
       
    def prepare_dim(self, x):
        return x.permute(0,2,1)[:,None, :,:]
    
    def _train_step(self, epoch, n_epochs):
        self.model.train()
        self.model.sample(True)
        elbos_train = []
        errors_train = []
        kl_list = []
        for batch_idx, train_data in enumerate(iter(self.train_dataloader)):
            
            x_train, y_train = train_data[0], train_data[1]
            y_train = torch.eye(self.model.n_classes)[y_train]
            x_train, y_train = self.prepare_dim(x_train).to(self.device), y_train.to(self.device)
            kl = 0.0
            logits = torch.zeros((self.nmc_train, x_train.shape[0], self.model.n_classes),
                             device=self.device)
            for s in range(self.nmc_train):
                logits[s, :, :], _kl = self.model.forward(x_train)
                kl += _kl
            kl = kl / self.nmc_train
            kl_list.append(kl.item())
            beta = metrics.get_beta(batch_idx-1, len(self.train_dataloader), self.beta_type, epoch, n_epochs)
            train_elbo = metrics.elbo(logits, y_train, kl, beta=beta, beta_type=self.beta_type, datasize=len(self.train_dataloader.dataset))
            train_error = self.compute_error(logits.mean(dim=0), y_train)
            if self.verbose == True:
                print('Train ELBO: {}, train error {} '.format(train_elbo.item(), train_error))
            self.optim.zero_grad()
            train_elbo.backward()
            self.optim.step()
            # self.scheduler.step()
            elbos_train.append(train_elbo.item()) 
            errors_train.append(train_error.item())
        return np.mean(elbos_train), np.mean(errors_train), np.mean(kl_list) 
    
    def fit(self, n_epochs, test_interval):
        self.model.to(self.device)
        losses_train, errors_train, losses_test, errors_test = [],[],[],[]
        for epoch in range(n_epochs):
            elbo_train, error_train, _ = self._train_step(epoch, n_epochs)    
            losses_train.append(elbo_train)
            errors_train.append(error_train)
            if epoch % test_interval == 0:
                test_loss, test_error = self.test()
                losses_test.append(test_loss.item()) 
                errors_test.append(test_error.item())
                print('TEST: epoch %d, nll %f, error %f' %(epoch, test_loss, test_error))
        return losses_train, errors_train, losses_test, errors_test
    
    

    
