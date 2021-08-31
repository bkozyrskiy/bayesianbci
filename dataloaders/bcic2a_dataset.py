
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys, os
from functools import reduce
from torch.utils.data.dataloader import DataLoader



RELATIVE_BRAINDECODE_PATH = '../../braindecode'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, RELATIVE_BRAINDECODE_PATH)))

from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess
from braindecode.datautil.windowers import create_windows_from_events

def get_dataloaders(subjects, batch_size, events=None, permute=None, insert_rgb_dim=False):
    '''
    Default dim order trials channels time
    '''
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subjects)
    low_cut_hz = 0.  # low cut frequency for filtering
    high_cut_hz = 38.  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000
    preprocessors = [
        # keep only EEG sensors
        MNEPreproc(fn='pick_types', eeg=True, meg=False, stim=False),
        # convert from volt to microvolt, directly modifying the numpy array
        NumpyPreproc(fn=lambda x: x * 1e6),
        # bandpass filter
        MNEPreproc(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
        # exponential moving standardization
        NumpyPreproc(fn=exponential_moving_standardize, factor_new=factor_new,
            init_block_size=init_block_size)
    ]
    preprocess(dataset, preprocessors)

    trial_start_offset_seconds = -0.5
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
    )

    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']
    valid_set = splitted['session_E']
    ood_dataloader = None
    
    if permute is None:
        permute = (0,1,2)
    
    tmp_train_data = list(zip(*[(torch.Tensor(elem[0][None,:,:]), torch.Tensor([elem[1]]).long()) for elem in train_set]))
    Xtr, ytr = torch.cat(tmp_train_data[0], dim=0), torch.cat(tmp_train_data[1], dim=0)
    
    tmp_val_data = list(zip(*[(torch.Tensor(elem[0][None,:,:]), torch.Tensor([elem[1]]).long()) for elem in valid_set]))
    Xval, yval = torch.cat(tmp_val_data[0], dim=0), torch.cat(tmp_val_data[1], dim=0)
    
    n_chans, input_time_length = Xtr.shape[1], Xtr.shape[2]
    
    Xtr, Xval = Xtr.permute(*permute), Xval.permute(*permute)
    
    if insert_rgb_dim:
        Xtr, Xval = Xtr[:,None,:,:], Xval[:,None,:,:] 
    
    if events is not None:
        tr_mask = reduce( (lambda x, y: x|y), [ytr == event  for event in events])
        Xtr, ytr = Xtr[tr_mask], ytr[tr_mask]
        val_mask = reduce((lambda x, y: x|y), [yval == event  for event in events])
        Xood, yood = Xval[torch.bitwise_not(val_mask)], yval[torch.bitwise_not(val_mask)]
        Xval, yval = Xval[val_mask], yval[val_mask]
        
    
    train_set = TensorDataset(Xtr, ytr)
    valid_set = TensorDataset(Xval, yval)
    
    
    # if events is not None:
    #     tmp_data = list(zip(*[(torch.Tensor(elem[0][None,:,:]),torch.Tensor([elem[1]]).long()) for elem in train_set if elem[1] in events]))
    #     train_set = TensorDataset(torch.cat(tmp_data[0], dim=0), torch.cat(tmp_data[1], dim=0))
        
    #     tmp_data = list(zip(*[(torch.Tensor(elem[0][None,:,:]),torch.Tensor([elem[1]]).long()) for elem in valid_set if elem[1] not in events]))
    #     ood_set = TensorDataset(torch.cat(tmp_data[0], dim=0), torch.cat(tmp_data[1], dim=0))
        
    #     tmp_data = list(zip(*[(torch.Tensor(elem[0][None,:,:]),torch.Tensor([elem[1]]).long()) for elem in valid_set if elem[1] in events]))
    #     valid_set = TensorDataset(torch.cat(tmp_data[0], dim=0), torch.cat(tmp_data[1], dim=0))
       
    if batch_size == None:
        batch_size_train = len(train_set)
        batch_size_test = len(valid_set)
    else:
        batch_size_train=batch_size
        batch_size_test=batch_size
    train_dataloader = DataLoader(train_set,batch_size=batch_size_train, shuffle=True,
                                    num_workers=0,
                                    pin_memory=False)
    test_dataloader = DataLoader(valid_set,batch_size=batch_size_test, shuffle=False,
									num_workers=0,
									pin_memory=False)
    if events is not None:
        ood_set = TensorDataset(Xood, yood)
        ood_dataloader = DataLoader(ood_set, batch_size=len(ood_set), shuffle=False,
                                        num_workers=0,
                                        pin_memory=False)
    else:
        ood_dataloader = None
    
    return train_dataloader, test_dataloader, ood_dataloader, n_chans, input_time_length


if __name__ == '__main__':
    train_dataloader, test_dataloader, ood_dataloader, n_chans, input_time_length =\
        get_dataloaders(subjects=[1], batch_size=None, events=[0,1], permute=(0,1,2), insert_rgb_dim=True)
    pass



