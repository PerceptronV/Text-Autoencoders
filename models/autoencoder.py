import tensorflow as tf
import os
import time
from tqdm import tqdm

from models.utils import Logger
from models.encoder import Encoder
from models.decoder import Decoder


class Autoencoder(tf.keras.Model):
  def __init__(self, encoder, decoder, batch_sz, rnn_units, start_tok, end_tok):
    super(Autoencoder, self).__init__(name='autoencoder')
    self.encoder = encoder
    self.decoder = decoder
    self.batch_sz = batch_sz
    self.rnn_units = rnn_units
    self.sos = start_tok
    self.eos = end_tok

  def compile(self, loss_function, optimizer):
    self.loss_func = loss_function
    self.optim = optimizer

  def set_ckpt_dir(self, path):
    self.ckpt_dir = path
    if not os.path.exists(self.ckpt_dir):
      os.mkdir(self.ckpt_dir)

  def train_step(self, src, targ, enc_hidden, training=True):
    with tf.GradientTape() as tape:
      enc_vec, _, enc_state = self.encoder(src, enc_hidden)
      dec_hidden = enc_state
      dec_input = tf.expand_dims([self.sos] * self.batch_sz, 1)

      loss = 0

      for t in range(1, targ.shape[1]):
        logits, _, dec_hidden = self.decoder(dec_input, enc_vec, dec_hidden)
        probs = tf.nn.softmax(logits)
        loss += self.loss_func(targ[:, t], probs)
        dec_input = tf.argmax(probs, axis=-1)

    batch_loss = loss / int(targ.shape[1])

    if training:
      variables = self.encoder.trainable_variables + self.decoder.trainable_variables
      gradients = tape.gradient(loss, variables)
      self.optim.apply_gradients(zip(gradients, variables))

    return tf.reduce_mean(batch_loss, axis=0)[0]

  def load_weights(self, epc):
    weights_dir = os.path.join(self.ckpt_dir, str(epc))
    enc_weights_dir = os.path.join(weights_dir,
                                   'encoder_weights.h5')
    dec_weights_dir = os.path.join(weights_dir,
                                   'decoder_weights.h5')
    self.encoder.load_weights(enc_weights_dir)
    self.decoder.load_weights(dec_weights_dir)

  def evaluate(self, dataset, progress=True):
    enc_hidden = tf.zeros((self.batch_sz, self.rnn_units))
    total_loss = 0
    steps = 0

    iter = tqdm(dataset) if progress == True else dataset

    for src, targ in iter:
      batch_loss = self.train_step(src, targ, enc_hidden, training=False)
      total_loss += batch_loss
      steps += 1
    return total_loss / steps

  def train(self, epochs, dataset, valset, benchmark, tokenizer):
    loaded = False
    logger = Logger(self.ckpt_dir)

    logger.log('Benchmark: {}\n\n'.format(
      tokenizer.sequences_to_texts([benchmark])[0]
    ), 'benchmark.txt', 'w')

    for ep in range(epochs):
      start = time.time()

      weights_dir = os.path.join(self.ckpt_dir,
                                 str(ep + 1))

      if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
      logfile = os.path.join(str(ep + 1),
                             'log.txt')

      enc_hidden = tf.zeros((self.batch_sz, self.rnn_units))
      total_loss = 0
      steps = 0

      enc_weights_dir = os.path.join(weights_dir,
                                     'encoder_weights.h5')
      dec_weights_dir = os.path.join(weights_dir,
                                     'decoder_weights.h5')

      if os.path.exists(enc_weights_dir) and os.path.exists(dec_weights_dir):
        logger.reprint(logfile)
        continue

      logger.log('EPOCH {} BEGINS'.format(ep + 1), logfile, 'w')

      last_enc_weights_dir = os.path.join(self.ckpt_dir,
                                          str(ep),
                                          'encoder_weights.h5')
      last_dec_weights_dir = os.path.join(self.ckpt_dir,
                                          str(ep),
                                          'decoder_weights.h5')
      if os.path.exists(last_enc_weights_dir) and os.path.exists(last_dec_weights_dir) and not loaded:
        print('Loading weights...')
        self.encoder.load_weights(last_enc_weights_dir)
        self.decoder.load_weights(last_dec_weights_dir)
        sample_out = self.call(
          tf.expand_dims(benchmark, 0),
          tf.zeros((1, self.rnn_units)),
          100
        )
        print('Sample output before training: {}'.format(
          tokenizer.sequences_to_texts([sample_out])[0]
        ))

      loaded = True

      for (batch, (src, targ)) in enumerate(dataset):
        batch_loss = self.train_step(src, targ, enc_hidden)

        total_loss += batch_loss
        steps += 1

        if (batch + 1) % 50 == 0:
          logger.log('Epoch {} Batch {} Loss {} Elapsed time {} sec'.format(
            ep + 1, batch + 1, batch_loss, time.time() - start
          ), logfile)

      sample_out = self.call(
        tf.expand_dims(benchmark, 0),
        tf.zeros((1, self.rnn_units)),
        100
      )

      self.encoder.save_weights(enc_weights_dir)
      self.decoder.save_weights(dec_weights_dir)

      logger.log('\nEpoch {} Loss {}'.format(
        ep + 1, total_loss / steps
      ), logfile)
      logger.log('Epoch {} Validation Loss {}'.format(
        ep + 1, self.evaluate(valset, progress=False).numpy()
      ), logfile)
      logger.log('Sample output after training: {}'.format(
        tokenizer.sequences_to_texts([sample_out])[0]
      ), logfile)
      logger.log('Time taken for epoch: {} sec\n'.format(
        time.time() - start
      ), logfile)

  def cycle(self, enc_vec, enc_state, maxlen):
    out = [self.sos]
    attn_weights = []
    step = 1

    dec_state = enc_state
    tok = self.sos
    dec_input = tf.expand_dims([self.sos], 1)

    while (tok != self.eos and tok != 0):
      if maxlen is not None:
        if step == maxlen:
          break

      logits, dec_attn, dec_state = self.decoder(dec_input, enc_vec, dec_state)
      dec_input = tf.argmax(tf.nn.softmax(logits), axis=-1)
      tok = dec_input[0][0].numpy()

      out.append(tok)
      attn_weights.append(dec_attn[0][0].numpy())
      step += 1

    return out, attn_weights

  def call(self, inp, enc_hidden, maxlen=None):
    enc_vec, _, enc_state = self.encoder(inp, enc_hidden)
    out, _ = self.cycle(enc_vec, enc_state, maxlen)

    return out
