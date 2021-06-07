from dataloaders.hgd_dataset import get_dataloaders
from models.layers import square, safe_log
import torch
from models.bayesian_model import LLBayesianShallowModel, Trainer
import os
import numpy as np


def all_subjects_experiment():
    subjects = range(1,15)
    device = 'cuda:0'
    model_checkpoints = 'checkpoints/'
    res = {}
    for subject in subjects:
        train_dataloader, test_dataloader, _, n_chans, input_time_length = get_dataloaders(subjects=[subject], batch_size=64, events=[0,1])
        priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.001),
                'posterior_rho_initial': (-3, 0.1),
        }
        model = LLBayesianShallowModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=2, conv_nonlin=square, pool_nonlin=safe_log,priors=priors)
        lr = 0.00625 
        n_epochs = 250
        nmc_train, nmc_test = 1, 1
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        trainer = Trainer(model, optim, train_dataloader=train_dataloader, beta_type="Blundell", nmc_train=nmc_train, nmc_test=nmc_test,
                      test_dataloader=test_dataloader, device=device, verbose=True)
        losses_train, errors_train, losses_test, errors_test = trainer.fit(n_epochs=n_epochs,test_interval=1)
        best_epoch = np.argmin(losses_test)
        res[subject] = errors_test[best_epoch]
        torch.save(model.state_dict(), os.path.join(model_checkpoints, 'subj{}.pth'.format(subject)))
    print('Bayesian results')
    for k in res:    
        print("Subject {}, lowest_error {}".format(k, res[k]))

def single_subject_exp(subject):
    device = 'cuda:0'
    batch_size = 128
    train_dataloader, test_dataloader, _, n_chans, input_time_length = get_dataloaders([subject], batch_size, events=[0,1])
    priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
    }
    model = LLBayesianShallowModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=2, conv_nonlin=square, pool_nonlin=safe_log, priors=priors)
    
    lr = 0.00625
    nmc_train, nmc_test = 1, 1
    n_epochs = 250
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    trainer = Trainer(model, optim, train_dataloader=train_dataloader, beta_type="Blundell", nmc_train=nmc_train, nmc_test=nmc_test,
                      test_dataloader=test_dataloader, device=device, verbose=True)
    trainer.fit(n_epochs=n_epochs,test_interval=1)
    pass

if __name__ == '__main__':
   all_subjects_experiment()