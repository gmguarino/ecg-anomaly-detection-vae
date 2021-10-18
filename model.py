"""
La classe torch.nn.Module viene usata per definire in modo generale un modello.
I module sono dei componenti che eseguono operazioni su dati tracciate dalla
funzione di autograd, salvando i gradienti. Un modello consiste di una estensione
della classe Module e può contenere varie componenti interne (tipo i layer)
che sono a loro volta estensioni di Module in modo che le operazioni che 
eseguono possano essere tracciate.
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    La classe Module si usa per costruire una rete neurale definendo due metodi
    principali: __init__ e forward.

    Per maggiori info: https://pytorch.org/docs/stable/generated/torch.nn.Module.html

    Il seguente modello è un 'encoder'. Si tratta di un modello che prende una
    sequenza di dati di input e la comprime in una sequenza più piccola (embedding 
    dimension).

    """

    def __init__(self, seq_len, n_features=1, embedding_dim=64):
        """
        Qui vengono definiti gli attributi della rete neurale o modello, tra cui 
        figurano i vari layer o addirittura alri modelli usati come componenti.
        """
        # Inizializza la classe superiore (nn.Module)
        super(LSTMEncoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        
        # definisce due strati LSTM
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        """
        Forward definisce le operazioni che il modello esegue su dei dati.
        """
        # reshape into (1, ecg_length, n_features=1)
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (embedding, _) = self.rnn2(x)

        # rehsape embedding into (n_features=1, embedding dimension)
        return embedding.reshape((self.n_features, self.embedding_dim))


class LSTMDecoder(nn.Module):
    """
    Questo modello esegue l'opposto dell'encoder. Si tratta si un 'decoder' che
    prende una rappresentazione compressa di una sequenza e prova a ricostruire
    la sequenza originale. 
    """

    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(LSTMDecoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x).T


class LSTMAutoencoder(nn.Module):
    """
    I due modelli qui sopra, l'encoder ed il decoder, vengono qui combinati in 
    un modello chiamato autoencoder. Questo modello prende come input una sequenza,
    la comprime, e poi prova a ricostruirla. Ciò ha due effetti:
    1. La rimozione del rumore dalla sequenza, visto il rumore viene perso durante
    la compressione.
    2. Il modello imparerà a replicare molto bene UN TIPO di segnale, mentre faticherà
    a riprodurre senza errore un segnale di tipo diverso, o comunque con meccanismi 
    diversi.

    Ciò lo rende un ottimo modello per la anomaly detection! Perché è resistente
    al rumore, ed è ottimo a riconoscere meccanismi anomali nei segnali.

    Per maggiori informazioni:
    
    - https://www.deeplearningitalia.com/introduzione-agli-autoencoder/

    - https://keras.io/examples/timeseries/timeseries_anomaly_detection/ (questo 
    usa keras/tensorflow invece di Pytorch ma è ottimo)
    """

    def __init__(self, seq_len, n_features, embedding_dim=64, device='cpu'):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = LSTMDecoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x