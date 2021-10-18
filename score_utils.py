import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from model import LSTMAutoencoder
from ecg_dataset import ECG5000


def get_scores(y_true, y_pred):
    """
    Funzione che calcola le metriche della performance del modello.

    Accuracy = Risultati Predetti Correttamente / Totale

    Per usare un linguaggio Covid-19 friendly:

    Recall = Veri Positivi / (Veri Positivi + Falsi Negativi) : Recupero
    Precision = Veri Positivi / (Veri Positivi + Falsi Positivi) : Precisione
    F1 = 2 * (Precision * Recall) / (Precision + Recall) : Media armonica di Precisione e Recupero
    
    Dove Positivo = Anomalia, Negativo = Normale 
    """
    results = {}
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['recall'] = recall_score(y_true, y_pred, average='micro')
    results['precision'] = precision_score(y_true, y_pred, average='micro')
    results['f1_score'] = f1_score(y_true, y_pred, average='micro')
    return results

def merge_scores(scores_1, scores_2):
    """
    Fuzione che fa la media degli score per i due dataset
    """
    results = {}

    keys = list(scores_1.keys())

    for key in keys:
        results[key] = (scores_1[key] + scores_2[key]) / 2

    return results

def print_results(results):
    """
    Stampo il dizionario dei risultati in formato di tabella
    """

    res_string = "\nStatistic \t Value\n----------\t------\n"
    for key, val in results.items():
        new_string = f"{key}   \t {val}\n"
        res_string += new_string
    
    print(res_string)


def evaluate(model, dataloader, threshold, criterion, device='cpu', test=True):
    """
    Valutazione del modello su un set di dati
    """
    # Inizializzo liste per tenere gli output e gli input/verità del modell
    predictions = []
    ground_truths = []
    # Iterando sui segnali
    for batch in dataloader:
        if test:
            # Se si tratta di un segnale con anomalia, viene messo 1 come verita
            signal, beat = batch
            gt = 1
        else:
            # Se no viene messo 0
            signal = batch
            gt = 0
        
        # La previsione del modello viene sempre effettuata nel contesto
        # torch.no_grad() per evitare di accumulare gradiente ed avere memory leak.
        with torch.no_grad():
            signal = signal.to(device)
            pred = model(signal)
            loss = criterion(torch.flatten(pred), torch.flatten(signal))
        
        # Se l'errore è sopra la soglia, classificare come anomalia
        if loss.item() > threshold:
            predictions.append(1)
        
        else: # se è sotto classificare come segnale normale
            predictions.append(0)
        ground_truths.append(gt)

    return np.array(predictions), np.array(ground_truths)


def score_on_ECG500(model, threshold, device):

    """
    Funzione per la valutazione del modello secondo più statistiche
    """
    # Prima per i segnali di tipo normale

    # Carico il Dataset in un DataLoader
    eval_dataset = ECG5000('data/ECG5000', phase='val')
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=True
    )

    # Specifico la loss
    eval_loss = nn.L1Loss(reduction='sum').to(device)
    # Valutaione modello
    predictions, ground_truths = evaluate(model, eval_loader, threshold, eval_loss, device=device, test=False)
    print(f"N of eval differences : {np.where(predictions != ground_truths)[0].size} / {len(eval_loader)}")

    # Analizzo risultati per il primo set di segnali
    eval_results = get_scores(ground_truths, predictions)
    

    # Ripeto per segnali con anomalie dovute ad aritmie
    test_dataset = ECG5000('data/ECG5000', phase='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True
    )

    test_loss = nn.L1Loss(reduction='sum').to(device)
    predictions, ground_truths = evaluate(model, test_loader, threshold, test_loss, device=device, test=True)

    print(f"N of test differences : {np.where(predictions != ground_truths)[0].size} / {len(test_loader)}")

    test_results = get_scores(ground_truths, predictions)

    #  ritorno entrambi i set di risultati
    return eval_results, test_results

