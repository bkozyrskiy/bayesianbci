import numpy as np
import matplotlib.pyplot as plt


def get_entropy_uncertainties(p_hat):
    pred_entropy = -(p_hat.mean(axis=0) * np.log(p_hat.mean(axis=0))).sum(axis=-1)
    aleatoric_unc = -((p_hat*np.log(p_hat)).sum(axis=-1)).mean(axis=0) 
    epistemic_unc = pred_entropy - aleatoric_unc
    confidence  = p_hat.mean(axis=0).max(axis=-1)
    return pred_entropy, epistemic_unc, aleatoric_unc, confidence

# def plot_historgrams()