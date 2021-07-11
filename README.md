# Text-Autoencoders

This repository contains the code for training two text autoencoders (English and Spanish) with TensorFlow 2.x

## Usage

## Pre-requisites

In order to run the training program, it is recommended to use **Python >=3.8**.

Additionally, the following libraries must be installed:

* **tensorflow >= 2.***

* nltk

* tqdm

* sklearn

* unicodedata

### Default Training

Training with the default configurations is very easy.

1. Clone this repository, then navigate into it.
   ```commandline
   git clone https://github.com/PerceptronV/Text-Autoencoders
   ```
   ```commandline
   cd Text-Autoencoders
   ```

2. Navigate into the `./training/` directory.
   ```commandline
   cd training
   ```
   
3. Run the main training program with default configurations:
   ```commandline
   python train.py
   ```

## Custom Parameters

Custom parameters can be specified in the command 
line as arguments, in the following format:

```commandline
python train.py [-h] [-se SEED] [-bs BATCHSIZE] [-st] [-ru RNNUNITS]
                [-eb EMBDIM] [-lr LEARNINGRATE] [-ep EPOCHS] [-ck CKPTDIR]

Train an English and Spanish text autoencoder.

optional arguments:
  -h, --help            show this help message and exit
  -se SEED, --seed SEED
                        Random seed (int)
  -bs BATCHSIZE, --batchsize BATCHSIZE
                        Batch size (int)
  -st, --stem           Stem words in preprocessing? (boolean)
  -ru RNNUNITS, --rnnunits RNNUNITS
                        Number of RNN units (int)
  -eb EMBDIM, --embdim EMBDIM
                        Embedding dimensions (int)
  -lr LEARNINGRATE, --learningrate LEARNINGRATE
                        Initial learning rate (float)
  -ep EPOCHS, --epochs EPOCHS
                        Number of epochs (int)
  -ck CKPTDIR, --ckptdir CKPTDIR
                        Checkpoint directory
```
