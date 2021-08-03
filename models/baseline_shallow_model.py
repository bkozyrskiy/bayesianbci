import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR



# import argparse

import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from utils import Expression, test_on_ood
from dataloaders.bcic2a_dataset import get_dataloaders
from models.layers import square, safe_log, AvgPool2dWithConv


class ShallowModel(nn.Module):
    '''Receives an input (trials x channels x time)'''
    
    def __init__(self, in_chans, input_time_length, n_classes, n_filters_time=40, 
                 filter_time_length=25, n_filters_spat=40, pool_time_length=75, pool_time_stride=15, conv_nonlin=None, pool_nonlin=None) -> None:
        super(ShallowModel, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module("conv_time", nn.Conv2d(1, n_filters_time, (filter_time_length , 1), stride=1))
        self.model.add_module("conv_spat", nn.Conv2d(n_filters_time, n_filters_spat, (1, in_chans), stride=1))
        self.model.add_module("conv_nonlin", Expression(conv_nonlin))
        self.model.add_module('pool', nn.AvgPool2d(kernel_size=(pool_time_length, 1),
                                                                stride=(pool_time_stride, 1)))
        self.model.add_module("pool_nonlin", Expression(pool_nonlin))
        out_feat_dim = self.model(torch.ones(1, 1, input_time_length, in_chans))
        self.final_conv_length = out_feat_dim.shape[2]
        self.model.add_module(
            "conv_classifier",
            nn.Conv2d(
                n_filters_spat,
                n_classes,
                (self.final_conv_length, 1),
                bias=True,
            ),
        )
        self.model.add_module('flatten',nn.Flatten())
    
    def forward(self, x):
        return self.model(x)
    



class Trainer():
    def __init__(self, model, optim, scheduler, train_dataloader, test_dataloader, device, verbose=False) -> None:
        self.model = model
        self.optim = optim
        self.device = device
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.verbose = verbose
        self.scheduler = scheduler
    
    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_loss, test_error = 0,0 
            for test_data in self.test_dataloader:
                x_test, y_test = test_data[0], test_data[1]
                x_test, y_test = self.prepare_dim(x_test).to(self.device), y_test.to(self.device)
                logits_test = self.model.forward(x_test)
                test_loss += nn.CrossEntropyLoss()(logits_test,y_test.long())
                # test_nll += -self.CE_likelihood(logits_test, y_test).sum(dim=1).mean(dim=0)
                test_error += self.compute_error(logits_test, y_test)
            test_loss /= len(self.test_dataloader)
            test_error /= len(self.test_dataloader)
        return test_loss, test_error   
    
    def compute_error(self,logits, y_true): 
        return (torch.argmax(logits,dim=1) != y_true).float().mean()
       
    def prepare_dim(self, x):
        return x.permute(0,2,1)[:,None, :,:]
    
    def fit(self, n_epochs, test_interval):
        self.model.to(self.device)
        losses_train, errors_train, losses_test, errors_test = [],[],[],[]
        for epoch in range(n_epochs):
            self.model.train()
            for train_data in iter(self.train_dataloader):
                x_train, y_train = train_data[0], train_data[1]
                x_train, y_train = self.prepare_dim(x_train).to(self.device), y_train.to(self.device)
                logits = self.model.forward(x_train)
                loss_func = nn.CrossEntropyLoss()
                train_loss = loss_func(logits, y_train.long())
                train_error = self.compute_error(logits, y_train)
                if self.verbose == True:
                    print('Train Loss: {}, train error {} '.format(train_loss.item(), train_error))
                self.optim.zero_grad()
                train_loss.backward()
                self.optim.step()
                self.scheduler.step()
                losses_train.append(train_loss.item()) 
                errors_train.append(train_error.item())
                
            if epoch % test_interval == 0:
                test_loss, test_error = self.test()
                losses_test.append(test_loss) 
                errors_test.append(test_error)
                print('TEST: epoch %d, loss %f, error %f' %(epoch, test_loss, test_error))
        return losses_train, errors_train, losses_test, errors_test

# def test_models(models, dataloader, n_classes, device):
#     for idx,model in enumerate(models):
#         model.eval()
#         model.to(device)
#         n_trials = len(dataloader.dataset)
#         probs_test = torch.empty(len(models),n_trials,n_classes)
#         with torch.no_grad():
#             test_loss, test_error = 0,0
#             start = 0 
#             y = torch.empty(n_trials, device=device)
#             for test_data in dataloader:
#                 x_batch, y_batch = test_data[0], test_data[1]
#                 x_batch = (x_batch.permute(0, 2, 1)[:,None, :,:]).to(device)
#                 y[start:start + x_batch.shape[0]] = y_batch
#                 probs_test[idx,start:start + x_batch.shape[0],:] = F.softmax(model.forward(x_batch))
#                 start = start + x_batch.shape[0]
                
#     torch.argmax(probs_test.mean(dim=0), dim=-1) != y 
#     return                

def all_subjects_experiment():
    subjects = range(1,10)
    mean_errors = []
    for subject in subjects:
        if not os.path.exists('checkpoints/bcic2a/subj{}'.format(subject)):
            os.makedirs('checkpoints/bcic2a/subj{}'.format(subject))
        models, mean_error = single_subject_exp(subject, n_epochs=500, n_ensambles=10, checkpoint_path='checkpoints/bcic2a/subj{}'.format(subject))
        mean_errors.append(mean_error)
    for subject in subjects:
        print('Subject {}, {:.2f}'.format(subject,mean_errors[subject-1]))
        
def single_subject_exp(subject, n_epochs=None, n_ensambles=1, checkpoint_path=None):
    device = 'cuda:0'
    # data_path = '/home/bogdan/ecom/BCI/datasets/bcic2A_mat/'
    # x,y = get_data(subject=9, training=True, PATH=data_path, order=('tr','t','ch'))
    # train_dataloader, test_dataloader = get_dataloaders(x, y, batch_size=64)
    # n_trials, input_time_length, n_chans, = x.shape
    train_dataloader, test_dataloader, _, n_chans, input_time_length = get_dataloaders(subjects=[subject], batch_size=64, events=[0,1])
    models = []
    models_test_errors = []
    for n_ensamble in range(n_ensambles):
        model = ShallowModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=2, conv_nonlin=square, pool_nonlin=safe_log)
        # out = model.forward(x[:,None,:,:])
        lr = 0.0625 * 0.01
        if n_epochs is None:
            n_epochs = 500
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optim, T_max=n_epochs-1)
        trainer = Trainer(model, optim, scheduler, train_dataloader,test_dataloader, device, verbose=False)
        losses_train, errors_train, losses_test, errors_test = trainer.fit(n_epochs=n_epochs, test_interval=1)
        models_test_errors.append(errors_test[-1].item())
        
        if checkpoint_path is not None:
           torch.save(model.state_dict(), os.path.join(checkpoint_path, '{}.pth'.format(n_ensamble)))
        models.append(model.to('cpu')) 
    
    mean_error = np.mean(models_test_errors)
    # print('Subject {}, {:.2f}'.format(subject,mean_error))
    return models, mean_error
 

     
 
if __name__ == "__main__":
    single_subject_exp(subject=1, n_epochs=None, n_ensambles=1, checkpoint_path=None)
    