import tensorflow as tf
import nltk

from sklearn.model_selection import train_test_split

import unicodedata
import re
import os
import io

# The following code is modified from https://www.tensorflow.org/text/tutorials/nmt_with_attention
# I added the option of stemming, filtering out of short-length sentences,
# Data provided by http://www.manythings.org/anki/

# Copyright 2019 The TensorFlow Authors, licensed under the Apache License, Version 2.0
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
                 if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w, lang, stem):
  if lang == 'en':
    stemmer = nltk.stem.snowball.SnowballStemmer("english")
  elif lang == 'sp':
    stemmer = nltk.stem.snowball.SnowballStemmer("spanish")

  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.strip()

  if stem:
    w = ' '.join([stemmer.stem(i) for i in w.split(' ')])

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '$ ' + w + ' &'
  return w


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, stem):
  map = {0: 'en', 1: 'sp'}

  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

  word_pairs = [[preprocess_sentence(w, map[e], stem)
                 for e, w in enumerate(line.split('\t'))
                 ]
                for line in lines
                # if len(line.split('\t')[0].split(' ')) > 5
                # Uncomment if you want to only include sentences with >5 words
                ]

  return zip(*word_pairs)


def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer


def load_dataset(path, num_examples=None, stem=False):
  # creating cleaned input, output pairs
  targ_lang, inp_lang = create_dataset(path, stem)

  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def to_tfds(input_tensor, label_tensor, buffer_sz, batch_sz, rand_seed):
  return tf.data.Dataset.from_tensor_slices(
      (input_tensor, label_tensor)
  ).shuffle(
      buffer_sz, seed=rand_seed
  ).batch(batch_sz, drop_remainder=True)


def load_data(batch_sz, rand_seed, stem):
  path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

  path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"

  sp_tensor, en_tensor, sp_lang, en_lang = load_dataset(path_to_file, stem)

  # Calculate max_length of the target tensors
  max_length_en, max_length_sp = en_tensor.shape[1], sp_tensor.shape[1]

  # Creating training and validation sets using an 80-20 split
  sp_tensor_train, sp_tensor_val, en_tensor_train, en_tensor_val = train_test_split(sp_tensor, en_tensor, test_size=0.2)

  vocab_sp_size = len(sp_lang.word_index) + 1
  vocab_en_size = len(en_lang.word_index) + 1

  # Show length
  print('Training set samples: {}'.format(len(sp_tensor_train)))
  print('Validation set samples: {}'.format(len(sp_tensor_val)))
  print('Spanish vocab size: {}; English vocab size: {}\n'.format(vocab_sp_size, vocab_en_size))

  buffer_size = int(len(sp_tensor_train) * 1.1)
  steps_per_epoch = len(sp_tensor_train) // batch_sz

  sp_train_dataset = to_tfds(sp_tensor_train, sp_tensor_train, buffer_size, batch_sz, rand_seed)
  sp_val_dataset = to_tfds(sp_tensor_val, sp_tensor_val, buffer_size, batch_sz, rand_seed)
  en_train_dataset = to_tfds(en_tensor_train, en_tensor_train, buffer_size, batch_sz, rand_seed)
  en_val_dataset = to_tfds(en_tensor_val, en_tensor_val, buffer_size, batch_sz, rand_seed)

  return (sp_train_dataset, sp_val_dataset, en_train_dataset, en_val_dataset,
          sp_tensor_train, en_tensor_train,vocab_sp_size, vocab_en_size,
          sp_lang, en_lang, steps_per_epoch)
