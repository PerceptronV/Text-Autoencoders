import tensorflow as tf
import numpy as np
import os

from dataloader import load_data
from models.encoder import Encoder
from models.decoder import Decoder
from models.autoencoder import Autoencoder

import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Train an English and Spanish text autoencoder.')

  parser.add_argument('-se', '--seed', type=int, default=7799, help="Random seed (int)")
  parser.add_argument('-bs', '--batchsize', type=int, default=64, help="Batch size (int)")
  parser.add_argument('-st', '--stem', action='store_true', help='Stem words in preprocessing? (boolean)')
  parser.add_argument('-ru', '--rnnunits', type=int, default=512, help="Number of RNN units (int)")
  parser.add_argument('-eb', '--embdim', type=int, default=256, help="Embedding dimensions (int)")
  parser.add_argument('-lr', '--learningrate', type=int, default=1e-3, help="Initial learning rate (float)")
  parser.add_argument('-ep', '--epochs', type=int, default=32, help="Number of epochs (int)")
  parser.add_argument('-ck', '--ckptdir', type=str, default='./.ckpts/', help="Checkpoint directory")

  args = parser.parse_args()

  PARAMS = {
    'RAND_SEED': args.seed,
    'RNN_UNITS': args.rnnunits,
    'EMB_DIM': args.embdim,
    'BATCH_SIZE': args.batchsize,
    'LR': args.learningrate,
    'AUTOENCODE_EPOCHS': args.epochs,
    'ROOT_CKPT_DIR': args.ckptdir
  }


  # Reproducible results
  tf.random.set_seed(PARAMS['RAND_SEED'])
  np.random.seed(PARAMS['RAND_SEED'])

  if not os.path.exists(PARAMS['ROOT_CKPT_DIR']):
    os.mkdir(PARAMS['ROOT_CKPT_DIR'])


  # Data loading

  (sp_train_dataset, sp_val_dataset, en_train_dataset, en_val_dataset,
   sp_tensor_train, en_tensor_train, vocab_sp_size, vocab_en_size,
   sp_lang, en_lang, steps_per_epoch) = load_data(PARAMS['BATCH_SIZE'], PARAMS['RAND_SEED'], args.stem)


  # Initialising models

  en_encoder = Encoder(vocab_en_size,
                       PARAMS['EMB_DIM'],
                       PARAMS['RNN_UNITS'])
  sp_encoder = Encoder(vocab_sp_size,
                       PARAMS['EMB_DIM'],
                       PARAMS['RNN_UNITS'])

  en_decoder = Decoder(vocab_en_size,
                       PARAMS['EMB_DIM'],
                       PARAMS['RNN_UNITS'])
  sp_decoder = Decoder(vocab_sp_size,
                       PARAMS['EMB_DIM'],
                       PARAMS['RNN_UNITS'])

  en_autoencoder = Autoencoder(
    en_encoder, en_decoder,
    PARAMS['BATCH_SIZE'], PARAMS['RNN_UNITS'],
    en_lang.word_index['$'], en_lang.word_index['&']
  )
  sp_autoencoder = Autoencoder(
    sp_encoder, sp_decoder,
    PARAMS['BATCH_SIZE'], PARAMS['RNN_UNITS'],
    sp_lang.word_index['$'], sp_lang.word_index['&']
  )

  sample_en_autoenc_inp = en_tensor_train[0]
  sample_en_autoenc_out = en_autoencoder(
    tf.expand_dims(sample_en_autoenc_inp, 0),
    tf.zeros((1, PARAMS['RNN_UNITS'])),
    100
  )
  print('Sample English input and output')
  print('Input: ' + en_lang.sequences_to_texts([sample_en_autoenc_inp])[0])
  print('Autoencoder output: ' + en_lang.sequences_to_texts([sample_en_autoenc_out])[0] + '\n')

  print('Sample Spanish input and output')
  sample_sp_autoenc_inp = sp_tensor_train[0]
  sample_sp_autoenc_out = sp_autoencoder(
    tf.expand_dims(sample_sp_autoenc_inp, 0),
    tf.zeros((1, PARAMS['RNN_UNITS'])),
    100
  )
  print('Input: ' + sp_lang.sequences_to_texts([sample_sp_autoenc_inp])[0])
  print('Autoencoder output: ' + sp_lang.sequences_to_texts([sample_sp_autoenc_out])[0] + '\n')


  # Loss function and optimizers

  # Learning rate scheduler
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    PARAMS['LR'],
    decay_steps=steps_per_epoch * 5,
    decay_rate=0.96,
    staircase=True
  )

  en_optim = tf.keras.optimizers.Adam(lr_schedule)
  sp_optim = tf.keras.optimizers.Adam(lr_schedule)

  crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none'
  )

  en_autoencoder.compile(crossentropy, en_optim)
  sp_autoencoder.compile(crossentropy, sp_optim)


  # Training

  en_autoencoder.set_ckpt_dir(os.path.join(
    PARAMS['ROOT_CKPT_DIR'], 'en_autoencoder'
  ))

  en_autoencoder.train(PARAMS['AUTOENCODE_EPOCHS'],
                       en_train_dataset,
                       en_val_dataset,
                       en_tensor_train[10],
                       en_lang)

  sp_autoencoder.set_ckpt_dir(os.path.join(
    PARAMS['ROOT_CKPT_DIR'], 'sp_autoencoder'
  ))

  sp_autoencoder.train(PARAMS['AUTOENCODE_EPOCHS'],
                       sp_train_dataset,
                       sp_val_dataset,
                       sp_tensor_train[10],
                       sp_lang)
