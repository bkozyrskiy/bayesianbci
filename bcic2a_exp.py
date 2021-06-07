from moabb_dataset import get_dataloaders
from layers import square, safe_log
import os
import torch
from bayesian_model import LLBayesianShallowModel, Trainer
import numpy as np

def all_subjects_experiment():
    subjects = range(1,10)
    device = 'cuda:0'
    model_checkpoints = 'checkpoints/'
    res = {}
    for subject in subjects:
        train_dataloader, test_dataloader, n_chans, input_time_length,_ = get_dataloaders(subjects=[subject], batch_size=64, events=[0,1])
        priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.001),
                'posterior_rho_initial': (-3, 0.1),
        }
        model = LLBayesianShallowModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=2, conv_nonlin=square, pool_nonlin=safe_log,priors=priors)
        lr = 0.0625 * 0.01
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
    # data_path = '/home/bogdan/ecom/BCI/datasets/bcic2A_mat/'
    # x,y = get_data(subject=9, training=True, PATH=data_path, order=('tr','t','ch'))
    # train_dataloader, test_dataloader = get_dataloaders(x, y, batch_size=64)
    # n_trials, input_time_length, n_chans, = x.shape
    train_dataloader, test_dataloader, n_chans, input_time_length, _ = get_dataloaders(subjects=[subject], batch_size=64, events=[0,1])
    priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
    }
    model = LLBayesianShallowModel(in_chans=n_chans, input_time_length = input_time_length, n_classes=2, conv_nonlin=square, pool_nonlin=safe_log, priors=priors)
    # out = model.forward(x[:,None,:,:])
    lr = 0.0625 * 0.01
    nmc_train, nmc_test = 1, 1
    n_epochs = 500
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = CosineAnnealingLR(optim, T_max=n_epochs-1)
    trainer = Trainer(model, optim, train_dataloader=train_dataloader, beta_type="Blundell", nmc_train=nmc_train, nmc_test=nmc_test,
                      test_dataloader=test_dataloader, device=device, verbose=True)
    trainer.fit(n_epochs=n_epochs,test_interval=1)
    pass
    # reliability_plot(model, test_dataloader, device)