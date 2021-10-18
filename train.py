import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import joblib
import matplotlib.pyplot as plt

from model import LSTMAutoencoder
from ecg_dataset import FakeECG, ECG5000
from train_utils import train_epoch
EPOCHS = 100

# Ci si basa sul seguente post per allenare un autoencoder che riconosca le anomalie in un ECG
# Poi proveremo ad applicare questo modello ad un altro dataset
# https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/

# Controllo se la GPU è disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training model on device: '{device}'")

# train_dataset = FakeECG(200, 140, fs=250, noise=0.1)
train_dataset = ECG5000('data/ECG5000') # Definiamo il dataset
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
) # Usiamo un data loader per passarlo al modello

model = LSTMAutoencoder(seq_len=140, n_features=1, embedding_dim=128, device=device)
# la lunghezza dei segmenti ecg è di 140 campioni, con un dimensione interna di 128
model.to(device) # Se diponibile il modello è trasferito su gpu

optimizer = optim.Adam(model.parameters()) # Ottimizzatore adam standard
# Se non migliorano le performance ridurre learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, mode='min', factor=0.5, verbose=True)

# Criterio di allenamento: L1Loss (Mean Absolute Error) ridotto come somma invece che come media - effetti locali più pronunciati -
criterion = nn.L1Loss(reduction='sum').to(device)

# Dizionario dove salvare i dati
history = dict(train=[], val=[])

#Ignora
loss_buffer = []

#Ignora
def process_loss(loss, loss_buffer, n_epochs=5, tol=0.001):
    return loss_buffer
    # if len(loss_buffer) < n_epochs:
    #     loss_buffer.append(loss)
    # else:
    #     loss_buffer.append(loss)
    #     loss_buffer.pop(0)

    #     if loss > min(loss_buffer) - tol:
    #         raise ValueError('Loss not improving')
    # return loss_buffer
    

# Allenamento del modello
for epoch in range(1, EPOCHS + 1):
    model.train() # Modello in modalità training
    try:
        # Calcolo loss e allenamento per una epoca
        loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch_n=epoch, verbose=True)
        # Passare il loss allo scheduler che decide se diminuire il learning rate
        scheduler.step(loss)
        # salvo dati
        history["train"].append(loss)
        # TODO : Add validation in training
        loss_buffer = process_loss(loss, loss_buffer, n_epochs=10, tol=0.001)
    except ValueError:
        print("Breaking due to loss not improving")
        break

# Salvo il modello 
torch.save(model.state_dict(), './cfg/model.pth')

# Salvo la il minimo della loss da usare come criterio di anomalia quando utilizzo il modello
joblib.dump(min(history["train"]), './cfg/min_loss.pkl')

# Plotto il grafico
plt.figure()
epochs = [i for i in range(EPOCHS)]
plt.plot(epochs, history["train"])
plt.xlabel("Epochs")
plt.ylabel("Loss: Mean Absolute Error")
plt.grid()
plt.show()
