import torch
import torch.nn.functional as F
import numpy as np


import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from uncertainty import get_entropy_uncertainties
from models.baseline_shallow_model import ShallowModel
from models.baseline_deep_model import DeepModel
from dataloaders.bcic2a_dataset import get_dataloaders
from models.layers import square, safe_log
from utils import suppress_stdout
from uncertainty_utils import get_model_predictions, get_error
 

def get_ensamble_uncertainties(subject, test_dataloader, ood_dataloader, events, n_chans, input_time_length, model_type, path_to_checkpoints=None, device='cuda:0'):
    if path_to_checkpoints is None:
        path_to_checkpoints = '../checkpoints/bcic2a_baseline/'
    
    model_dir = os.path.join(path_to_checkpoints, 'subj{}'.format(subject))
    models_files = os.listdir(model_dir)
    test_predictions = np.empty((len(models_files), len(test_dataloader.dataset), len(events)))
    ood_predictions = np.empty((len(models_files), len(ood_dataloader.dataset), len(events)))
    for m_idx, model_file in enumerate(models_files):
        if model_type == 'shallow':
            model = ShallowModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=len(events), conv_nonlin=square, pool_nonlin=safe_log)
        elif model_type == 'deep':
            model = DeepModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=2, final_conv_length="auto").create_network()
        else:
            raise ValueError("unrecognised model type")    
        model.load_state_dict(torch.load(os.path.join(model_dir,model_file)))
        test_predictions[m_idx,:,:], y_true = get_model_predictions(model, test_dataloader, n_classes=len(events), device=device)
        ood_predictions[m_idx,:,:], _ = get_model_predictions(model, ood_dataloader, n_classes=len(events), device=device)
    
    error = get_error(test_predictions, y_true)
    test_entropy, test_epistemic, test_aleatoric, test_confidence = get_entropy_uncertainties(test_predictions)
    ood_entropy, ood_epistemic, ood_aleatoric, ood_confidence = get_entropy_uncertainties(ood_predictions)
    
    return (test_entropy, test_epistemic, test_aleatoric, test_confidence), (ood_entropy, ood_epistemic, ood_aleatoric, ood_confidence), error
    

if __name__ == "__main__":
    device = "cuda:0"
    path_to_checkpoints = 'checkpoints/bcic2a_baseline/'
    subjects = range(1,10)
    events = [0,1]
    for subject in subjects:
        with suppress_stdout():
            train_dataloader, test_dataloader, ood_dataloader, n_chans, input_time_length = get_dataloaders(subjects=[subject], batch_size=64, events=events)
        model_dir = os.path.join(path_to_checkpoints, 'subj{}'.format(subject))
        models_files = os.listdir(model_dir)
        test_predictions = np.empty((len(models_files), len(test_dataloader.dataset), len(events)))
        ood_predictions = np.empty((len(models_files), len(ood_dataloader.dataset), len(events)))
        for m_idx, model_file in enumerate(models_files):
            model = ShallowModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=len(events), conv_nonlin=square, pool_nonlin=safe_log)
            model.load_state_dict(torch.load(os.path.join(model_dir,model_file)))
            test_predictions[m_idx,:,:], y = get_model_predictions(model, test_dataloader, n_classes=len(events), device=device)
            ood_predictions[m_idx,:,:], _ = get_model_predictions(model, ood_dataloader, n_classes=len(events), device=device)
        test_error = get_error(test_predictions, y)
        test_entropy, test_epistemic, test_aleatoric, test_confidence = get_entropy_uncertainties(test_predictions)
        ood_entropy, ood_epistemic, ood_aleatoric, ood_confidence = get_entropy_uncertainties(ood_predictions)
        print('Subject {}, test error {:.2f}, entropy {:.2f}, epistemic {:.2f}, aleatoric {:.2f}'.format(subject, test_error, test_entropy.mean(), 
                                                                                            test_epistemic.mean(), test_aleatoric.mean()))
        print('OOD entropy {:.2f}, epistemic {:.2f}, aleatoric {:.2f}'.format(ood_entropy.mean(), ood_epistemic.mean(), ood_aleatoric.mean()))
        print(' ')
        pass
