# Adversarial-attack-on-CAN-IDS

This repository contains the source code of adversarial attack on CAN IDS.

We consider 3 attacks types (Dos, fuzzy, spoofing)

Dataset
=================
In order to download data, you need to access the URL https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset and download it via email.

Training and Testing
============================
To train a model, `run train.py`. The model will be checkpointed (saved) after each epoch to the networks/models directory.

For example, to train a Inception_Resnet with 200 epochs and a batch size of 128:

    python train.py --model Inception_Resnet --epochs 200 --batch_size 128
    
To perform attack, `run attack.py`. To specify the types of models to test, `use --model`.

    python attack.py --model Inception_Resnet
