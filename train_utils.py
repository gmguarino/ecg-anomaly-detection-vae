import torch
import math
import time

def train_batch(model, batch, optimizer, loss_criterion):
    """
    Funzione che allena il modello per un batch
    """
    prediction = model(batch) # Applicazione del modello al batch di dati
    loss = loss_criterion(prediction.unsqueeze(0), batch) # predizione dell'errore
    optimizer.zero_grad() # Rimuovere gradienti di operazioni precedenti
    loss.backward() # Generare nuovi gradienti
    optimizer.step() # Aggiornamento parametri

    return loss.item() # Return solo del valore di loss, non del tensore

def train_epoch(model, data_loader, optimizer, loss_criterion, device, epoch_n=1, verbose=True):
    """
    Allenamento del modello per una epoca
    """
    loss = 0 # Mettere il loss a 0 per iniziare
    t = time.time() # Inizializzare un contatore per verificare le performance
    for batch in data_loader: # Iterazione dei batch nel DataLoader
        # counter += 1
        # print(counter)
        batch = batch.to(device) # Utilizza la GPU se disponibile
        loss += train_batch(model, batch, optimizer, loss_criterion) # Aggiungere al loss totale
    
    n_batches = math.ceil(len(data_loader.dataset) / data_loader.batch_size) # Calcolo del numero di batch 
    if verbose:
        # Se verbose == True, stampare dettagli sull'allenamento
        print(f"Epoch {epoch_n}; Loss: {loss / n_batches}; Time Taken: {time.time() - t}")
    return loss / n_batches # ritorno del loss medio