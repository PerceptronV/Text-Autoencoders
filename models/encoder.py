import tensorflow as tf


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, emb_dim, rnn_units):
    super(Encoder, self).__init__(name='encoder')
    self.embed = tf.keras.layers.Embedding(vocab_size,
                                           emb_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=False,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.map = tf.keras.layers.Dense(rnn_units)

  def call(self, inputs, hidden):
    x = self.embed(inputs)
    out, state = self.gru(x, initial_state=hidden)
    return self.map(out), out, state
