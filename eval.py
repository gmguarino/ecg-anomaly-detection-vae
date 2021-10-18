import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from model import LSTMAutoencoder
from ecg_dataset import FakeECG, ECG5000




def get_device():

    return torch.device("cuda" if torch.cuda.is_available() else "cpu") # Se disponibile usa GPU

# Valutazione del modello sulle anomalie


def get_model(device=torch.device("cpu")):
    # Definizione del modello e spostamento su GPU
    model = LSTMAutoencoder(seq_len=140, n_features=1, embedding_dim=128, device=device)
    model.to(device)
    # Carico parametri modello
    model.load_state_dict(torch.load('./cfg/model.pth'))
    return model


def evaluate_model(test=True, device=torch.device("cpu")):
    """
    Questa funzione applica il modello a nuovi dati per ispezionarne il funzionamento
    """
    # Se la fase è test il modello riceverà delle anomalie dovute ad aritmie
    if test:
        phase = 'test'
    # Se la fase è 'val', il modello riceverà segnali normali che dovrà riprodurre precisamente
    else:
        phase = 'val'
    dataset = ECG5000('data/ECG5000', phase=phase) # Definisco dataset
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    ) # Uso un dataloader per passare i dati
    iter_loader = iter(loader) # Uso il loader come un iteratore
    if test:
        # Prendo un campione
        batch, arr = next(iter_loader)
    else:
        batch = next(iter_loader)

    with torch.no_grad(): 
        # Questo contesto viene usato come protezione per 
        #evitare un possibile calcolo dei gradienti nei meccanismi interni

        # Sposto i dati a GPU e passo per il modello
        batch = batch.to(device)
        pred = model(batch)
        # Calcolo la differenza tra verità e segnale predetto
        loss = criterion(torch.flatten(pred), torch.flatten(batch))

    return batch, pred, loss


def check_for_anomaly(loss, threshold):
    # Controllo se il valore è superiore al threshold consentito
    if loss > threshold:
        return 'Anomaly'
    else:
        return 'Normal'


if __name__=="__main__":

    n_tests = 4

    device = get_device()

    model = get_model(device)
    # Modello in modalità valutazione.. non aggiorna i parametri e/o non accumula gradienti
    model.eval()

    criterion = nn.L1Loss(reduction='sum').to(device) # Definisco il criterio di performance


    # Threshold = 2 x loss minima del processo di allenamento
    threshold = joblib.load('./cfg/min_loss.pkl') * 2

    for t_idx in range(n_tests):

        test = True if t_idx%2 else False

        # Valuto su segnale normale
        batch, pred, loss = evaluate_model(test=test, device=device)

        # Passo dati e predizioni a CPU e li rendo array piatte
        pred = pred.cpu().numpy().flatten()
        batch = batch.cpu().numpy().flatten()

        # Plotto i riusultati a confronto
        plt.figure()
        plt.plot(pred, label='Predicted')
        plt.plot(batch, label='Ground Truth')
        plt.xlabel("Sample Number")
        plt.ylabel("Signal (arb. units)")
        gt = "Anomaly" if test else "Normal"
        plt.title(gt + " | Predicted: " + check_for_anomaly(loss.item(), threshold) + f" with score {loss.item():.2f} / {threshold:.2f}")
        plt.legend()

    # # Ripeto per segnale con aritmia
    # batch, pred, loss = evaluate_model(test=True, device=device)

    # pred = pred.cpu().numpy().flatten()
    # batch = batch.cpu().numpy().flatten()

    # plt.figure(2)
    # plt.plot(pred, label='Predicted')
    # plt.plot(batch, label='Ground Truth')
    # plt.xlabel("Sample Number")
    # plt.ylabel("Signal (arb. units)")
    # plt.title("Anomaly | Predicted: " + check_for_anomaly(loss.item(), threshold) + f" with score {loss.item():.2f} / {threshold:.2f}")
    # plt.legend()

    plt.show()