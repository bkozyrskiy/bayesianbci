import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_bcic2a_data(subject,training,PATH, order=None):
	'''	Loads the dataset 2a of the BCI Competition IV
	available on http://bnci-horizon-2020.eu/database/data-sets
	Keyword arguments:
	subject -- number of subject in [1, .. ,9]
	training -- if True, load training data
				if False, load testing data
	
	Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
			class_return 	numpy matrix 	size = NO_valid_trial
	'''
	NO_channels = 22
	NO_tests = 6*48 	
	Window_Length = 7*250 

	class_return = np.zeros(NO_tests)
	data_return = np.zeros((NO_tests,NO_channels,Window_Length))

	NO_valid_trial = 0
	if training:
		a = sio.loadmat(PATH+'A0'+str(subject)+'T.mat')
	else:
		a = sio.loadmat(PATH+'A0'+str(subject)+'E.mat')
	a_data = a['data']
	for ii in range(0,a_data.size):
		a_data1 = a_data[0,ii]
		a_data2=[a_data1[0,0]]
		a_data3=a_data2[0]
		a_X 		= a_data3[0]
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
		a_fs 		= a_data3[3]
		a_classes 	= a_data3[4]
		a_artifacts = a_data3[5]
		a_gender 	= a_data3[6]
		a_age 		= a_data3[7]
		for trial in range(0,a_trial.size):
			if(a_artifacts[trial]==0):
				data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
				class_return[NO_valid_trial] = int(a_y[trial])
				NO_valid_trial +=1

	data_return, class_return = data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]
	if order is not None:
		default_order = ('tr','ch','t')
		permute_mask = []
		for target_key in order:
			permute_mask.append(default_order.index(target_key)) 		
		data_return = data_return.transpose(permute_mask)

	return data_return, class_return



def get_dataloaders(x,y,batch_size,random_state=0):
    y = y - 1 # for make class lables 0,1,2,3 instead of 1,2,3,4 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)
    X_train, X_test, y_train, y_test = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(y_train), torch.Tensor(y_test)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset,
									batch_size=batch_size,
									shuffle=True,
									num_workers=0,
									pin_memory=False)
    
    test_dataloader = DataLoader(test_dataset,
									batch_size=batch_size,
									shuffle=True,
									num_workers=0,
									pin_memory=False)
    return train_dataloader, test_dataloader
	

if __name__ == '__main__':
    data_path = '/home/bogdan/ecom/BCI/bcic2A/'
    # subjects = ['A0%dT.mat' %i for i in range(1,10)]
    x,y = get_data(1,training=True,PATH=data_path)
    
    pass