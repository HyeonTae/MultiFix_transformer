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
Etc. (Included in "requirements.txt")
- torch
- torchtext
- numpy
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

## HOW TO EXECUTE OUR MODEL?
## Data Processing
Download the dataset that has already been preprocessed
```
    $ bash download_processing_data.sh
```

## Model training
Train the data with our model.
```
    $ cd model
    $ cd train.py
```
