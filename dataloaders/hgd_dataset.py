import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import logging
from collections import OrderedDict

import os, sys
sys.path.append('/home/bogdan/ecom/BCI/braindecode_old/')
from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import highpass_cnt, bandpass_cnt
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.splitters import split_into_two_sets

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

def load_bbci_data(filename, low_cut_hz, high_cut_hz=None, debug=False):
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)

    log.info("Loading data...")
    cnt = loader.load()

    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                  clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))

    # now pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)

    # Further preprocessings as descibed in paper
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    if high_cut_hz is None:
        cnt = mne_apply(
            lambda a: highpass_cnt(
                a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
            cnt)
    else:
        cnt = mne_apply(
            lambda a: bandpass_cnt(
                a, low_cut_hz, high_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
            cnt)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    # Trial interval, start at -500 already, since improved decoding for networks
    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset


def load_train_valid_test(
        train_filename, test_filename, low_cut_hz, high_cut_hz, debug=False):
    log.info("Loading train...")
    full_train_set = load_bbci_data(
        train_filename, low_cut_hz=low_cut_hz, high_cut_hz=high_cut_hz, debug=debug)

    log.info("Loading test...")
    test_set = load_bbci_data(
        test_filename, low_cut_hz=low_cut_hz, high_cut_hz=high_cut_hz, debug=debug)
    # valid_set_fraction = 0.8
    # train_set, valid_set = split_into_two_sets(full_train_set,
    #                                            valid_set_fraction)

    log.info("Train set with {:4d} trials".format(len(full_train_set.X)))
    # if valid_set is not None:
    #     log.info("Valid set with {:4d} trials".format(len(valid_set.X)))
    log.info("Test set with  {:4d} trials".format(len(test_set.X)))

    # return train_set, valid_set, test_set
    return full_train_set, test_set

def get_dataloaders(subjects, batch_size, events=None):
    data_folder = '/home/bogdan/ecom/BCI/datasets/high-gamma-dataset/data/'
    Xs_train, ys_train, Xs_test, ys_test = [],[],[],[]
    for subject in subjects:
        train_filename = os.path.join(data_folder, 'train','{}.mat'.format(subject))
        test_filename = os.path.join(data_folder, 'test','{}.mat'.format(subject))
        low_cut_hz = 4
        high_cut_hz = 38
        train_set, test_set = load_train_valid_test(train_filename, test_filename, low_cut_hz, high_cut_hz, debug=False)
        Xs_train.append(train_set.X), ys_train.append(train_set.y) 
        Xs_test.append(test_set.X), ys_test.append(test_set.y)
    X_train = np.concatenate(Xs_train,axis=0)
    y_train = np.concatenate(ys_train, axis=0)
    X_test = np.concatenate(Xs_test, axis=0)
    y_test = np.concatenate(ys_test,axis=0)
      
    
    if batch_size == None:
        batch_size_train = len(train_set)
        batch_size_test = len(test_set)
    else:
        batch_size_train=batch_size
        batch_size_test=batch_size
    if events is None:
        train_dataloader = DataLoader(TensorDataset(X_train, y_train),
                                batch_size=batch_size_train,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=False)
        test_dataloader = DataLoader(TensorDataset(X_test, y_test),
                                batch_size=batch_size_test,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=False)
        ood_dataloader = None
    else:
        idx_train = np.isin(y_train,events)
        train_dataloader = DataLoader(TensorDataset(torch.tensor(X_train[idx_train]), 
                                                    torch.tensor(y_train[idx_train], dtype=torch.long)),
                                                    batch_size=batch_size_train,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    pin_memory=False)
        idx_test = np.isin(test_set.y, events)
        test_dataloader = DataLoader(TensorDataset(torch.tensor(X_test[idx_test]),
                                                   torch.tensor(y_test[idx_test], dtype=torch.long)),
                                                batch_size=batch_size_test,
                                                shuffle=True,
                                                num_workers=0,
                                                pin_memory=False)
        idx_ood = np.logical_not(np.isin(test_set.y, events))
        ood_dataloader = DataLoader(TensorDataset(torch.tensor(X_test[idx_ood]), 
                                                  torch.tensor(y_test[idx_ood], dtype=torch.long)),
                                                batch_size=int(idx_ood.sum()),
                                                shuffle=True,
                                                num_workers=0,
                                                pin_memory=False)
    
    n_chans = train_set.X.shape[1]
    input_time_length = train_set.X.shape[2]
    return train_dataloader, test_dataloader, ood_dataloader, n_chans, input_time_length
    
    

if __name__ == '__main__':
    subj = 1
    data_folder = '/home/bogdan/ecom/BCI/datasets/high-gamma-dataset/data/'
    train_filename = os.path.join(data_folder, 'train','{}.mat'.format(subj))
    test_filename = os.path.join(data_folder, 'test','{}.mat'.format(subj))
    low_cut_hz = 4
    train_set, test_set = load_train_valid_test(train_filename, test_filename, low_cut_hz, debug=False)
    pass
