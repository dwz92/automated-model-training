'''
@Author: Raphael Fox, Chen Wu, Qi Er Teng
@GitHub: dwz92
@Discription: This script reconstruct the collective work of Raphael, Chen, and Qi Er in chen_250302_5E_16x1024_withoutSwlfreq3638.ipynb
into an instantiable NN class in Python.
'''


#### WORKPLACE SETUP ####
import pandas as pd
import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy.io import savemat
##########################



class AoADataset(Dataset):
    def __init__(self, filename, header=['mapped_freq', 'r2', 'i2', 'r3', 'i3', 'r4', 'i4', 'r5', 'i5', 'az_rad', 'el_rad']):
        super().__init__()

        self.df = pd.read_csv(filename)
        
        self.x = torch.tensor(self.df[header[:9]].values, dtype=torch.float32)
        self.y = torch.tensor(self.df[header[-2:]].values, dtype=torch.float32)

        # print(self.x)
        # print(self.y)

    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


class aoaMLP(nn.Module):
    def __init__(self, features, hiddens:list, output):
        super(aoaMLP, self).__init__()

        #Construct Model
        layers = []
        in_size = features

        for h in hiddens:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h

        layers.append(nn.Linear(in_size, output))
        
        self.model = nn.Sequential(*layers)

        # self.fc1 = nn.Linear(features, hiddens[0])
        # self.fc2 = nn.Linear(hiddens[0], hiddens[1])
        # self.fc3 = nn.Linear(hiddens[1], hiddens[2])
        # self.fc4 = nn.Linear(hiddens[2], hiddens[3])
        # self.fc5 = nn.Linear(hiddens[3], hiddens[4])
        # self.fc6 = nn.Linear(hiddens[4], hiddens[5])
        # self.fc7 = nn.Linear(hiddens[5], hiddens[6])
        # self.fc8 = nn.Linear(hiddens[6], hiddens[7])
        # self.fc9 = nn.Linear(hiddens[7], hiddens[8])
        # self.fc10 = nn.Linear(hiddens[8], hiddens[9])
        # self.fc11 = nn.Linear(hiddens[9], hiddens[10])
        # self.fc12 = nn.Linear(hiddens[10], hiddens[11])
        # self.fc13 = nn.Linear(hiddens[11], hiddens[12])
        # self.fc14 = nn.Linear(hiddens[12], hiddens[13])
        # self.fc15 = nn.Linear(hiddens[13], hiddens[14])
        # self.fc16 = nn.Linear(hiddens[14], hiddens[15])
        # self.out = nn.Linear(hiddens[15], output)

        # self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        # x = self.relu(self.fc4(x))
        # x = self.relu(self.fc5(x))
        # x = self.relu(self.fc6(x))
        # x = self.relu(self.fc7(x))
        # x = self.relu(self.fc8(x))
        # x = self.relu(self.fc9(x))
        # x = self.relu(self.fc10(x))
        # x = self.relu(self.fc11(x))
        # x = self.relu(self.fc12(x))
        # x = self.relu(self.fc13(x))
        # x = self.relu(self.fc14(x))
        # x = self.relu(self.fc15(x))
        # x = self.relu(self.fc16(x))
        # x = self.out(x)
        x = self.model(x)

        return x
    


def AOAtrain(model: aoaMLP, K: int, freq_band: str, SNR: int,
             save_to: str, src_dir: str, 
             nepochs=2000, batchSize=512*2, learningRate=1e-4, MINIUMVALIDLOSS=1e-5, reference_channel=1,
             loss=nn.MSELoss, opt=optim.Adam,
             device='cpu'):
    
    # Path to train + valid datasets and loss output + model file
    for_test_file = f'{save_to}/FORTESTING_Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_SNR{SNR}_withoutSwl.pth'
    net_file_name = f'{save_to}/Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_SNR{SNR}_withoutSwl.pth'
    Loss_file_name = f'{save_to}/Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_SNR{SNR}_withoutSwl_Loss.mat'
    trainName = f'{src_dir}/Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_train_SNR{SNR}_withoutSwl.csv'
    validName = f'{src_dir}/Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_valid_SNR{SNR}_withoutSwl.csv'

    # net_file_name = os.path.join(save_to, f'Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_SNR{SNR}_withoutSwl.pth')
    # Loss_file_name = os.path.join(save_to, f'Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_SNR{SNR}_withoutSwl_Loss.mat')
    # trainName = os.path.join(src_dir, f'Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_train_SNR{SNR}_withoutSwl.csv')
    # validName = os.path.join(src_dir, f'Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_valid_SNR{SNR}_withoutSwl.csv')

    # convert to absolute paths
    # net_file_name = os.path.abspath(net_file_name)
    # Loss_file_name = os.path.abspath(Loss_file_name)
    # trainName = os.path.abspath(trainName)
    # validName = os.path.abspath(validName)

    # print(validName)

    # Init datasets for training and validation
    traindata = AoADataset(filename=trainName)
    validata = AoADataset(filename=validName)

    trainLoader =  DataLoader(traindata, batch_size=batchSize, shuffle=True)
    validLoader = DataLoader(validata, batch_size=batchSize, shuffle=True)


    # Load model if exists
    if os.path.exists(net_file_name):
        model.load_state_dict(torch.load(net_file_name, weights_only=True))
        print(f"Loaded model from {net_file_name}")

    # Send to device
    model.to(device)

    # Define opt
    optimizer = opt(model.parameters(), lr=learningRate)

    # Loss Func
    lossFunc = loss()

    # Learning rate scheduler (decay lr when loss plateaus)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=20, threshold=1e-5, min_lr=1e-8)

    # Init parameters to save the best model
    bestVLoss = np.inf
    bestEpoch = 0
    bestWeights = None
    bestcount = 0


    # Training loop
    avgTLoss = 0.0
    batchLoss = 0.0
    stopCount = 0
    minDelta = 1e-9
    trainingLosses = []
    vLosses = []
    prevVLoss = np.inf
    nTotalSteps = len(trainLoader)

    for epoch in range(nepochs):
        # print(f"Epoch {epoch+1}")
        model.train()
        runningLoss = 0.0
        runningVLoss = 0.0
        for i, (data, targets) in enumerate(trainLoader):
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            pred = model(data)
            loss = lossFunc(pred, targets)
            runningLoss += loss.item()
            batchLoss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

        # Saves the average training loss over the entire epoch for graphing
        avgTLoss = runningLoss / nTotalSteps
        trainingLosses.append(avgTLoss)

        # Validation
        model.eval()
        with torch.no_grad():
            for v, (vData, vTargets) in enumerate(validLoader):
                vData = vData.to(device)
                vTargets = vTargets.to(device)
                vPred = model(vData)
                vLoss = lossFunc(vPred, vTargets)
                runningVLoss += vLoss.item()

        avgVLoss = runningVLoss / (len(validLoader))
        scheduler.step(avgVLoss)
        vLosses.append(avgVLoss)
        #print(f"Epoch {epoch+1}, Avg Train Loss = {avgTLoss:.10f}, Avg Valid Loss = {avgVLoss:.10f}, Learning Rate = {scheduler.get_last_lr()[-1]}\n")

        # Track best
        if avgVLoss < bestVLoss - minDelta:
            bestVLoss = avgVLoss
            bestEpoch = epoch + 1
            print(f"FreqBand_{freq_band}GHz_SNR{SNR}: BestEpoch = {bestEpoch:.10f}, Best valid Loss = {bestVLoss:.10f}, Learning Rate = {scheduler.get_last_lr()[-1]}\n")
            torch.save(model.state_dict(), net_file_name)
            stopCount = 0
            savemat(Loss_file_name, {'trainingLosses': trainingLosses, 'vLosses': vLosses})

            bestcount += 1

            if 0 < avgVLoss < MINIUMVALIDLOSS and bestcount > 3:
                print(f"FreqBand_{freq_band}GHz_SNR{SNR}:  MINIUMVALIDLOSS met. Training Terminating.")
                break
        else:
            stopCount += 1
        if stopCount >= 25:
            print(f"Stopping at epoch {epoch + 1}")
            break
    
    
    torch.save(model, for_test_file)
    
    print(f"FreqBand_{freq_band}GHz_SNR{SNR}: Lowest validation loss was {bestVLoss:.10f} in epoch {bestEpoch}")


def runTrain(freq_band: str, SNR: int,
             save_to: str, src_dir: str, K=5, 
             hidden=[1024, 1024, 1024, 1024,
             1024, 1024, 1024, 1024,
             1024, 1024, 1024, 1024,
             1024, 1024, 1024, 1024],
             reference_channel=1, output=2,
             device='cpu'
            ):
    try: 
        print("Starting "+ f'Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_SNR{SNR}_withoutSwl.pth')
        features = 1 + 2*(K-1)
        model = aoaMLP(features=features, hiddens=hidden, output=output)

        AOAtrain(model, K, nepochs=2000, freq_band=freq_band, SNR=SNR, save_to=save_to, src_dir=src_dir,
                        reference_channel=reference_channel, device=device)
        print(f'Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_SNR{SNR}_withoutSwl.pth Finished training.')
    except KeyboardInterrupt:
        print("Keyboard Interruption running " + f'Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_SNR{SNR}_withoutSwl.pth')
    except Exception as e:
        print(f"Error in {freq_band} {SNR}db: {e}")
    print(f'Ele{K}_RefCh{reference_channel}_FreqBand_{freq_band}GHz_SNR{SNR}_withoutSwl.pth Graceful Exist')



if __name__ == '__main__':
    hidden=[1024, 1024, 1024, 1024,
             1024, 1024, 1024, 1024,
             1024, 1024, 1024, 1024,
             1024, 1024, 1024, 1024]
    
    k = 5
    features = 1 + 2*(k-1)
    freq_band='2_4'
    SNR=10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    aoamodel = aoaMLP(features=features, hiddens=hidden, output=2)
    

    # AOAtrain(aoamodel, k, nepochs=2000, freq_band=freq_band, SNR=SNR, save_to='models_losses/', src_dir='datasets/',
    #                     reference_channel=1, device=device)

