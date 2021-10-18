import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from model import LSTMAutoencoder
from ecg_dataset import FakeECG, ECG5000
from score_utils import score_on_ECG500, merge_scores, print_results


# Questo script serve ad ottenere statistiche sul modello

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Passo a GPU se disponibile

print("Device in use: {}".format(device))

# Definisco modello e device
model = LSTMAutoencoder(seq_len=140, n_features=1, embedding_dim=128, device=device)
model.to(device)

# Carico parametri e metto in modalità valutazione
model.load_state_dict(torch.load('./cfg/model.pth'))
model.eval()

def check_for_anomaly(loss, threshold):
    if loss > threshold:
        return 'Anomaly'
    else:
        return 'Normal'

# Threshold = 2 x loss minima del processo di allenamento
threshold = joblib.load('./cfg/min_loss.pkl') * 2

# Valuto modello sia su anomalie che su segnali normali
val_results, test_results = score_on_ECG500(model, threshold, device)

# Unisco gli score
results = merge_scores(val_results, test_results)

# Stampo statistiche
print_results(results)