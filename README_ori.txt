# MultiFix: Learning to Repair Multiple Errors by Optimal Alignment Learning

## Overview
This project is a transformers implementation which learning to repair multiple errors by optimal alignment learning.

## Hardware
The models are trained using folloing hardware:
- Ubuntu 18.04.5 LTS
- NVIDA TITAN Xp 12GB
- Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz
- 64GB RAM

## Dependencies
- Python version is 3.6.7
We use the following version of Pytorch.
gpu support (CUDA==10.1)
- torch==1.1.0
gpu support (CUDA>10.1)
- torch==1.5.0
Etc. (Included in "requirements.txt")
- torchtext==0.3.1
- numpy==1.16.1
- tqdm
- matplotlib
- regex
- transformers

## Prerequisite
- Use virtualenv
```	sh
    sudo apt-get install build-essential libssl-dev libffi-dev python-dev
    sudo apt install python3-pip
    sudo pip3 install virtualenv
    virtualenv -p python3 venv
    . venv/bin/activate
    # code your stuff
    deactivate
```

## Datasets
Our dataset is based on the dataset provided by DeepFix.
https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip

## HOW TO EXECUTE OUR MODEL?
## Data Processing
Generate training data based on the DeepFix and DrRepair dataset.
```	sh
    bash data_processing.sh
```

## Model training
Train the data with our model.
```	sh
    bash model_training.sh
```

However, this takes a significant time, so we provide 2 models that were trained.
> log/pth
