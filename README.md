# Anomaly Detection of ECG segments

The model was developed in order to recognize anomalies in ECG segments due to misplacements of electrodes, contact noise and, in the worst case scenario, arrhythmias. <br/>
The architecture used is an LSTM Autoencoder learns to reproduce the shape of an ECG segment with a certain accuracy, defined by its loss (in this case an L1 loss was used). 

## Setting up the training
Firstly install some basic dependencies listed in the requirements file:
```
pip install -r requirements.txt
```
Then set up pytorch.
### If you have an NVIDIA CUDA GPU:
Check that you have the correct drivers and the CUDA Toolkit installed and, if not, install them. I have the CUDA Toolkit v10.2 which can be setup as per the [documentation](https://docs.nvidia.com/cuda/archive/10.2/). <br/>
You can install GPU supported PyTorch with
```
pip3 install torch torchvision torchaudio
```
Alternatively you can install both the toolkit and the pytorch using Anaconda:
```
conda install pytorch torchvision torchaudio cudatoolkit=<VERSION> -c pytorch
```
For more information on setup, visit the [PyTorch website](https://pytorch.org/get-started/locally/).
## Download the Data
The dataset used is the ECG 5000 dataset for outlier/anomaly detection. It can be downloaded at [paperswithcode.com](https://paperswithcode.com/sota/outlier-detection-on-ecg5000). Place the files in `data/ECG5000/`.
```
data
└── ECG5000
    ├── ECG5000_TEST.arff
    ├── ECG5000_TEST.ts
    ├── ECG5000_TEST.txt
    ├── ECG5000_TRAIN.arff
    ├── ECG5000_TRAIN.ts
    ├── ECG5000_TRAIN.txt
    └── ECG5000.txt
```

## Train and evaluate the model
The model is trained using the `train.py` script (genius), and can be scored against a series of metrics with the `score.py` script. The `eval.py` script evaluates 4 random examples from the dataset (2 'normal' ECGs and 2 'anomalous' ECGs) and plots the ecg segment together with the model's reconstructions. Captioned with the ground truth and the classification assigned.

## Examples of ECG classification

