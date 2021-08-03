
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import init
from torch.nn.functional import elu


# import argparse

import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from utils import Expression, test_on_ood, suppress_stdout
from dataloaders.bcic2a_dataset import get_dataloaders
from models.layers import square, safe_log, identity, AvgPool2dWithConv

class DeepModel(nn.Module):
    '''Receives an input (trials x 1 x time x channels)'''
    def __init__(self,
        in_chans,
        n_classes,
        input_time_length,
        final_conv_length,
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        n_filters_2=50,
        filter_length_2=10,
        n_filters_3=100,
        filter_length_3=10,
        n_filters_4=200,
        filter_length_4=10,
        first_nonlin=elu,
        first_pool_mode="max",
        first_pool_nonlin=identity,
        later_nonlin=elu,
        later_pool_mode="max",
        later_pool_nonlin=identity,
        drop_prob=0.5,
        double_time_convs=False,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        stride_before_pool=False     
    ):
        super(DeepModel,self).__init__()
        if final_conv_length == "auto":
            assert input_time_length is not None
        self.__dict__ = {}
        self.__dict__.update(locals())
        
    def create_network(self):
        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.pool_time_stride
        pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
        first_pool_class = pool_class_dict[self.first_pool_mode]
        later_pool_class = pool_class_dict[self.later_pool_mode]
        model = nn.Sequential()
        if self.split_first_layer:
            
            model.add_module(
                "conv_time",
                nn.Conv2d(
                    1,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                ),
            )
            model.add_module(
                "conv_spat",
                nn.Conv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    (1, self.in_chans),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            model.add_module("dimshuffle", Expression(_transpose_time_to_spat))
            model.add_module(
                "conv_time",
                nn.Conv2d(
                    self.in_chans,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            model.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv,
                    momentum=self.batch_norm_alpha,
                    affine=True,
                    eps=1e-5,
                ),
            )
        model.add_module("conv_nonlin", Expression(self.first_nonlin))
        model.add_module(
            "pool",
            first_pool_class(
                kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)
            ),
        )
        model.add_module("pool_nonlin", Expression(self.first_pool_nonlin))

        def add_conv_pool_block(
            model, n_filters_before, n_filters, filter_length, block_nr
        ):
            suffix = "_{:d}".format(block_nr)
            model.add_module("drop" + suffix, nn.Dropout(p=self.drop_prob))
            model.add_module(
                "conv" + suffix,
                nn.Conv2d(
                    n_filters_before,
                    n_filters,
                    (filter_length, 1),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            if self.batch_norm:
                model.add_module(
                    "bnorm" + suffix,
                    nn.BatchNorm2d(
                        n_filters,
                        momentum=self.batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    ),
                )
            model.add_module("nonlin" + suffix, Expression(self.later_nonlin))

            model.add_module(
                "pool" + suffix,
                later_pool_class(
                    kernel_size=(self.pool_time_length, 1),
                    stride=(pool_stride, 1),
                ),
            )
            model.add_module(
                "pool_nonlin" + suffix, Expression(self.later_pool_nonlin)
            )

        add_conv_pool_block(
            model, n_filters_conv, self.n_filters_2, self.filter_length_2, 2
        )
        add_conv_pool_block(
            model, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3
        )
        add_conv_pool_block(
            model, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4
        )

        # model.add_module('drop_classifier', nn.Dropout(p=self.drop_prob))
        model.eval()
        if self.final_conv_length == "auto":
            # out = model(
            #     torch.Tensor(
            #         np.ones(
            #             (1, self.in_chans, self.input_time_length, 1),
            #             dtype=np.float32,
            #         )
            #     )
            # )
            out = model(
                torch.Tensor(
                    np.ones(
                        (1, 1, self.input_time_length, self.in_chans),
                        dtype=np.float32,
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        model.add_module(
            "conv_classifier",
            nn.Conv2d(
                self.n_filters_4,
                self.n_classes,
                (self.final_conv_length, 1),
                bias=True,
            ),
        )
        model.add_module("softmax", nn.LogSoftmax(dim=1))
        model.add_module("squeeze", Expression(_squeeze_final_output))

        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        init.xavier_uniform_(model.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(model.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(model.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(model.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(model.bnorm.weight, 1)
            init.constant_(model.bnorm.bias, 0)
        param_dict = dict(list(model.named_parameters()))
        for block_nr in range(2, 5):
            conv_weight = param_dict["conv_{:d}.weight".format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)
            if not self.batch_norm:
                conv_bias = param_dict["conv_{:d}.bias".format(block_nr)]
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict["bnorm_{:d}.weight".format(block_nr)]
                bnorm_bias = param_dict["bnorm_{:d}.bias".format(block_nr)]
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)

        init.xavier_uniform_(model.conv_classifier.weight, gain=1)
        init.constant_(model.conv_classifier.bias, 0)

        # Start in eval mode
        model.eval()
        return model
    
    def forward(self, x):
        return self.model(x)

def _transpose_time_to_spat(x):
    # return x.permute(0, 3, 2, 1)
    return x.permute(0, 1, 3, 2)

def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


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
    
def single_subject_exp(subject, n_epochs=None, n_ensambles=1, checkpoint_path=None):
    device = 'cuda:0'
    # data_path = '/home/bogdan/ecom/BCI/datasets/bcic2A_mat/'
    # x,y = get_data(subject=9, training=True, PATH=data_path, order=('tr','t','ch'))
    # train_dataloader, test_dataloader = get_dataloaders(x, y, batch_size=64)
    # n_trials, input_time_length, n_chans, = x.shape
    with suppress_stdout():
        train_dataloader, test_dataloader, _, n_chans, input_time_length = get_dataloaders(subjects=[subject], batch_size=64, events=[0,1])
    models = []
    models_test_errors = []
    for n_ensamble in range(n_ensambles):
        model = DeepModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=2, final_conv_length="auto").create_network()
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
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            torch.save(model.state_dict(), os.path.join(checkpoint_path, '{}.pth'.format(n_ensamble)))
        models.append(model.to('cpu')) 
    
    mean_error = np.mean(models_test_errors)
    # print('Subject {}, {:.2f}'.format(subject,mean_error))
    return models, mean_error


if __name__ == '__main__':
    mean_errors = []
    for subject in range(1,10):
        checkpoint_path = 'checkpoints/bcic2a_deep_baseline/subj{}'.format(subject)
        models, mean_error = single_subject_exp(subject=subject, n_epochs=None, n_ensambles=10, checkpoint_path=checkpoint_path)
        mean_errors.append(mean_error)
    
    for subj in range(1,10):
        print("subj {}, mean test error {}".format(subj, mean_errors[subj]))
    