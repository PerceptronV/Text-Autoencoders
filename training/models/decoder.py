import tensorflow as tf


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, emb_dim, rnn_units):
    super(Decoder,self).__init__(name='decoder')
    self.emb_dim = emb_dim
    self.embed = tf.keras.layers.Embedding(vocab_size,
                                           emb_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.dense = tf.keras.layers.Dense(vocab_size)

    self.attn = tf.keras.layers.Dense(1, activation='sigmoid')

  def attention(self, vector, hidden):
    conc = tf.concat([vector, hidden], axis = -1)
    vector_weight = self.attn(conc)
    hidden_weight = 1 - vector_weight
    return vector_weight, hidden_weight

  def call(self, inp, vector, hidden):
    x = self.embed(inp)
    vector_weight, hidden_weight = self.attention(vector, hidden)
    state = vector_weight * vector + hidden_weight * hidden
    x, state = self.gru(x, initial_state = state)
    x = self.dense(x)
    return x, vector_weight, state
