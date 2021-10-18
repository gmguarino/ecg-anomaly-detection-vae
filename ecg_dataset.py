import torch
from torch.utils.data import Dataset, DataLoader

import os

import neurokit2 as nk
import numpy as np
import pandas as pd
from time import sleep
from scipy.io import arff
import matplotlib.pyplot as plt

from timeout import timeout


class ECG5000(Dataset):
    """
    La classe dataset viene estesa per fornire in modo efficiente i dati ad un 
    modello. Ha tre metodi principali, __init__, __len__ e __get_item__.
    """
    def __init__(self, root, phase='train'):
        """
        Qui vengono definiti gli attributi di istanza del Dataset
        """
        self.root = root
        self.phase = phase
        with open(os.path.join(self.root, 'ECG5000_TRAIN.arff')) as f:
            dataset1, meta1 = arff.loadarff(f)
        with open(os.path.join(self.root, 'ECG5000_TEST.arff')) as f:
            dataset2, meta2 = arff.loadarff(f)
        dataset = pd.concat([pd.DataFrame(dataset1), pd.DataFrame(dataset2)])
        dataset["target"] = pd.to_numeric(dataset["target"])
        if phase == 'train':
            dataset = dataset.loc[dataset['target'] == 1].iloc[:-200]
        elif phase == 'val':
            dataset = dataset.loc[dataset['target'] == 1].iloc[-200:]
        else:
            dataset = dataset.loc[dataset['target'] != 1]
        self.dataset = dataset

        super(ECG5000, self).__init__()
    
    def __len__(self):
        """
        Ha la funzione di definire la quantità di dati in un dataset, così che il 
        Dataloader sa quando finisce una epoca e vanno re-indicizzati e mischiati
        i dati.
        """
        return self.dataset.shape[0]

    def __getitem__(self, index):
        """
        Questa invece è la funzione che fa il lavoro pesante. Estrae i dati che 
        servono e li passa al modello.
        """
        ecg = self.dataset.loc[:, self.dataset.columns != 'target'].iloc[index]
        beat = self.dataset.loc[:, self.dataset.columns == 'target'].iloc[index].values
        ecg = pd.to_numeric(ecg)
        tensor = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0)
        
        if self.phase == "train" or self.phase == "val":
            return tensor
        else:
            return tensor, beat
        
        

class FakeECG(Dataset):
    """
    Class that generates a dataset of fake ECG signals to perform timeseries analysis.
    DEPRECATO NON USARE. Genera segnali ECG artificiali ma con performance instabili.

    """
    def __init__(self, n_signals, signal_length, fs=100, noise=0.1, anomalies=False):
        super(FakeECG, self).__init__()
        d = locals()
        for key, value in d.items():
            setattr(self, key, value)

    def __len__(self):
        return self.n_signals

    def __getitem__(self, index):
        hr = np.random.uniform(low=45, high=110)
        offset = np.random.randint(low=self.signal_length // 4, high=self.signal_length // 4 + self.signal_length // 10)
        # duration = l / sampling_rate
        ecg  = np.empty((1, self.signal_length))
        try:
            with timeout(1):
                ecg_temp = nk.ecg_simulate(
                        length=self.signal_length + offset, 
                        sampling_rate=self.fs, 
                        heart_rate=hr,
                        noise=self.noise,
                        # method='simple'
                )[offset:]
                noise = np.random.normal(scale=0.3, size=self.signal_length) * ecg_temp.max() * self.noise

                ecg[0, :] = FakeECG.normalize(
                    ecg_temp + noise
                )
                tensor = torch.tensor(ecg, dtype=torch.float32)
                return tensor
        except TimeoutError:
            return self.__getitem__(index)

    
    @staticmethod
    def normalize(ecg):
        return (ecg - ecg.mean()) / (ecg.max() - ecg.min())
        
if __name__ == '__main__':
    """
    Qui viene testato il dataset
    """
    ds = ECG5000(root='data/ECG5000', train=True)
    ds.__getitem__(2)
    train_dataset = FakeECG(100, 100, noise=0.01)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1
    )
    i = iter(train_dataset)
    batch = next(i)
    plt.figure()
    plt.plot(batch.numpy().flatten())
    # plt.show()