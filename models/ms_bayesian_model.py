import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import os,sys
sys.path.append('/home/bogdan/ecom/bayesianize')
import bnn

from baseline_model import ShallowModel
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from dataloaders.bcic2a_dataset import get_dataloaders
from models.layers import square, safe_log
from utils import suppress_stdout

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
                train_ce = loss_func(logits, y_train.long())
                kl = sum(m.kl_divergence() for m in self.model.modules() if hasattr(m, "kl_divergence"))
                train_loss = train_ce + kl / len(self.train_dataloader.dataset)
                
                train_error = self.compute_error(logits, y_train)
                if self.verbose == True:
                    print('Train Loss: {:.2f}, train CE {:.2f}, train error {:.2f} '.format(train_loss.item(), train_ce.item(), train_error.item()))
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
        

def all_subjects_experiment(model_checkpoints_path):
    subjects = range(1,10)
    test_errors = []
    for subject in subjects:
        model, mean_error = single_subject_exp(subject, n_epochs=150)
        test_errors.append(mean_error)
        torch.save(model.state_dict(), os.path.join(model_checkpoints_path, 'subj{}.pth'.format(subject)))
    for subject in subjects:
        print('Subject {}, {:.2f}'.format(subject,test_errors[subject-1]))
    
    
def single_subject_exp(subject, n_epochs=None):
    device = 'cuda:1'
    with suppress_stdout():
        train_dataloader, test_dataloader, _, n_chans, input_time_length = get_dataloaders(subjects=[subject], batch_size=64, events=[0,1])
    model = ShallowModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=2, conv_nonlin=square, pool_nonlin=safe_log)
    # bnn.bayesianize_(model, inference={model.model[0]:{'inference':'inducing', 'inducing_rows':20, 'inducing_cols':20,'prior_sd':1, 'sqrt_width_scaling':True},
    #                                    model.model[-2]: {'inference':'ffg'}})
    # bnn.bayesianize_(model, inference={model.model[0]:{'inference':'inducing', 'inducing_rows':20, 'inducing_cols':20,'prior_sd':1,'sqrt_width_scaling':True},
    #                                    model.model[1]:{'inference':'inducing', 'inducing_rows':20, 'inducing_cols':20,'prior_sd':1,'sqrt_width_scaling':True},
    #                                    model.model[-2]: {'inference':'ffg'}})
    bnn.bayesianize_(model, inference={model.model[0]:{'inference':'ffg','prior_weight_sd':1},
                                       model.model[1]:{'inference':'ffg','prior_weight_sd':1},
                                       model.model[-2]: {'inference':'ffg'}})
    lr = 0.000625
    if n_epochs is None:
        n_epochs = 150
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optim, T_max=n_epochs-1)
    trainer = Trainer(model, optim, scheduler, train_dataloader,test_dataloader, device, verbose=True)
    losses_train, errors_train, losses_test, errors_test = trainer.fit(n_epochs=n_epochs, test_interval=1)
    
    # print('Subject {}, {:.2f}'.format(subject,errors_test[-1]))
    return model, errors_test[-1]


if __name__ == '__main__':
    model_checkpoints_path='./checkpoints/ms_bayes_ffg/'
    # all_subjects_experiment(model_checkpoints_path=model_checkpoints_path)
    subject = 5
    model, error_test = single_subject_exp(subject=subject, n_epochs=100)
    torch.save(model.state_dict(), os.path.join(model_checkpoints_path, 'subj{}.pth'.format(subject)))