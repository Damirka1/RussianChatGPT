import tensorflow as tf
from tensorflow import keras
from tensorflow import math, range, pow, int32, int64, newaxis
from keras.models import Sequential
from keras.layers import Layer, Input, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention
from positional_encodings.tf_encodings import TFPositionalEncoding1D
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.metrics import SparseCategoricalCrossentropy

from utilities import PositionalEncoding, MaskedSelfAttention, create_masks

import numpy as np

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

load_md = load_model('model')

# importing navec - russian GloVe embedding model
from navec import Navec

# big pre trained model
path_model = 'navec_hudlit_v1_12B_500K_300d_100q.tar'

navec = Navec.load(path_model)

# load navec as gensim model
gs = navec.as_gensim

# add to gensim tag words
gs.add_vector('<start>', np.full((300,), -1))
gs.add_vector('<end>', np.full((300,), 1))

# Define hyperparameters
max_word_len = 300
embedding_dim = 300
num_heads = 4
dff = 64
dropout_rate = 0.1
learning_rate = 0.01
epochs = 100
num_layers = 4

# Define vocabulary size
vocab_size = len(navec.vocab.words) + 2

# Define input layer
# inputs = Input(shape=(max_len,), dtype=keras.int32)
encoder_inputs = Input(shape=(max_word_len,), dtype=int64)
decoder_inputs = Input(shape=(max_word_len,), dtype=int64)
# decoder_outputs = Input(shape=(max_word_len, vocab_size))

# # Add embedding layer
# embed = Embedding(input_dim=vocab_size, 
#                   output_dim=embedding_dim, 
#                   weights=[navec.get_weights()],
#                   trainable=False)(encoder_inputs)

# # # Add positional encoding layer
# embed = TFPositionalEncoding1D(embedding_dim)(embed)

# pos_embed = embed

# Add embedding layers to convert input and output sequences to dense vectors
input_embedded = PositionalEncoding(max_word_len, embedding_dim)(Embedding(input_dim=vocab_size, 
                                            output_dim=embedding_dim, 
                                            weights=[gs.vectors],
                                            trainable=False)(encoder_inputs))

output_embedded = PositionalEncoding(max_word_len, embedding_dim)(Embedding(input_dim=vocab_size, 
                                            output_dim=embedding_dim, 
                                            weights=[gs.vectors],
                                            trainable=False)(decoder_inputs))

# input_embedded = embed
# output_embedded = embed

# Add encoder layers
encoder_output = input_embedded

# Add encoder layers
for i in range(num_layers):
    # Add multi head Attention Layer
    attn = MultiHeadAttention(embedding_dim, num_heads)(encoder_output, encoder_output)
    attn = Dropout(dropout_rate)(attn)
    attn = LayerNormalization(epsilon=1e-6)(encoder_output + attn)
    
    # Add Feed Forward NN
    ff = Dense(dff, activation='relu')(attn)
    ff = Dense(embedding_dim)(ff)
    ff = Dropout(dropout_rate)(ff)
    
    encoder_output = LayerNormalization(epsilon=1e-6)(attn + ff)


# Add decoder layers
decoder_output = output_embedded
for i in range(num_layers):
    # Adding Masked self attention layer
    # masked_self_attn = MultiHeadAttention(embedding_dim, num_heads, mask=True)(decoder_output, decoder_output)
    masked_self_attn = MaskedSelfAttention(embedding_dim, num_heads)(decoder_output, decoder_output)
    masked_self_attn = LayerNormalization(epsilon=1e-6)(decoder_output + masked_self_attn)
    
    # Add multi head Attention Layer
    attn = MultiHeadAttention(embedding_dim, num_heads)(masked_self_attn, encoder_output)
    attn = Dropout(dropout_rate)(attn)
    attn = LayerNormalization(epsilon=1e-6)(masked_self_attn + attn)
    
    # Add Feed Forward NN
    ff = Dense(dff, activation='relu')(attn)
    ff = Dense(embedding_dim)(ff)
    ff = Dropout(dropout_rate)(ff)
    
    decoder_output = LayerNormalization(epsilon=1e-6)(attn + ff)


# Add output layer
outputs = Dense(units=vocab_size, activation='softmax')(decoder_output)

# Create model
# model = Model(inputs=[encoder_inputs, decoder_inputs, decoder_outputs], outputs=outputs)
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

model.set_weights(load_md.get_weights())

# print(model.summary())

text = input("Введите текст: ")

import dataset as ds

promt = ds.create_from_promt(text, gs)
print(promt.shape)

prediction = ds.get_start_prediction(gs)

print(prediction.shape)

res = model(inputs=[promt, prediction], training=False)

# print(res)

r = ds.read_from_pred(res, gs)

print(r)

