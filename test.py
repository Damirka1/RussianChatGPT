import numpy as np

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow import math, range, pow, int32, int64, newaxis
from keras.models import Sequential
from keras.layers import Layer, Input, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention
from positional_encodings.tf_encodings import TFPositionalEncoding1D
from keras.optimizers import Adam
from keras.models import Model
from keras.metrics import SparseCategoricalCrossentropy

from utilities import PositionalEncoding, MaskedSelfAttention, create_masks, CustomSchedule

# from utilities import *

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# importing navec - russian GloVe embedding model
from navec import Navec

from razdel import tokenize

# big pre trained model
path_model = 'navec_hudlit_v1_12B_500K_300d_100q.tar'

navec = Navec.load(path_model)

# load navec as gensim model
gs = navec.as_gensim
# print(gs.key_to_index['привет'],  gs.index_to_key[335377])

# add to gensim tag words
gs.add_vector('<start>', np.full((300,), -1))
gs.add_vector('<end>', np.full((300,), 1))

# print tags as vectors
# print(gs.get_vector('<start>'))
# print(gs.get_vector('<end>'))

# Define hyperparameters
max_word_len = 300
embedding_dim = 300
num_heads = 6
dff = 8192
dropout_rate = 0.5
# learning_rate = 0.1
epochs = 100
num_layers = 16

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

# Compile model
# model.compile()

# print(model.output_shape)
# print(model.summary())

# exit()

# training

from dataset import create_dataset, is_end_index, is_pad_index, is_unk_index, get_start_prediction

train_dataset = create_dataset('df (3).txt', gs, max_word_len);
# train_dataset = create_dataset('train.txt', gs, max_word_len);

# Define loss function and optimizer
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
learning_rate = CustomSchedule(embedding_dim)
optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')                 
                    
import time

# train_step_signature = [
#     tf.TensorSpec(shape=(max_word_len,), dtype=tf.int64),
#     tf.TensorSpec(shape=(max_word_len,), dtype=tf.int64),
# ]

# TEST 
# @tf.function(input_signature=train_step_signature)
# def train_step(inp, tar):
#   prediction = get_start_prediction()
#   correct_prediction = get_start_prediction()
  
#   # skip 1 token <start>
#   for i in range(1, len(tar)):
#     target = tar[i]
#     # print(inp.shape, prediction.shape)
#     print("Current word: ", gs.index_to_key[target])
    
#     pred_index = 0
    
#     for j in range(1):
#       with tf.GradientTape(persistent=True) as tape:
#         pred = model(inputs=[inp, correct_prediction], training=True)
#         loss = loss_function(target, pred[0][i-1])
#         # loss = loss_function(tar, pred[0]) # wrong - always predict <pad>
#         # if(get_end_index(target)):
#         #   loss = loss_function(tar, pred[0])
    
#       gradients = tape.gradient(loss, model.trainable_variables)
#       optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#       pred_index = np.argmax(pred[0][i-1].numpy())
    
#     pred_word = gs.index_to_key[pred_index]
    
#     prediction[i] = pred_index
#     correct_prediction[i] = target
    
#     print("Current predict: ", pred_index, pred_word)
#     print("Current predictions dict: ", sep='', end='')
#     for word in prediction[:i+1]:
#       print(gs.index_to_key[word], sep='', end=' ')
      
#     print("\nCorrect predictions dict: ", sep='', end='')
#     for word in correct_prediction[:i+1]:
#       print(gs.index_to_key[word], sep='', end=' ')
#     print("\n====================")
    
#     # print(np.expand_dims(target, axis=0).shape, np.expand_dims(prediction, axis=0).shape)
    
#     if(get_end_index(target)):
#       print("return")
#       break
    
#   # gradients = tape.gradient(loss, model.trainable_variables)
#   # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
#   train_loss(loss)
#   train_accuracy(accuracy_function(tar, pred))

train_step_signature = [
    tf.TensorSpec(shape=(max_word_len), dtype=tf.int64),
    tf.TensorSpec(shape=(max_word_len), dtype=tf.int64),
    tf.TensorSpec(shape=(max_word_len), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def step(inp, tar, prediction):
  with tf.GradientTape() as tape:
    pred = model(inputs=[inp, prediction], training=True)
    loss = loss_object(tar, pred)
  
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  train_loss(loss)
  train_accuracy(accuracy_function(tar, pred))

  return pred


def train_step(inp, tar):
  prediction = get_start_prediction(gs)
  
  # prediction = []
  
  # for idx in tar:
  #   if(is_end_index(idx)):
  #     break
    
  #   prediction.append(idx)
  
  # prediction = np.array(prediction)
  
  l = len(tar)
  if(l >= max_word_len):
    l = max_word_len - 1
  
  for i in range(1, l):
    if(is_end_index(tar[i])):
      break
    
    prediction[i] = tar[i]
    
  pred = step(inp, tar, prediction)
  
  print("True: ", sep='', end='')
  for word in prediction:
    print(gs.index_to_key[word], sep='', end=' ')
    
  print("\nPred: ", sep='', end='')
  for words in pred[0]:
    word = np.argmax(words.numpy())
    print(gs.index_to_key[word], sep='', end=' ')
  print()
  
  print(f'Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

# # @tf.function(input_signature=train_step_signature)
# def train_step(inp, tar):
#   with tf.GradientTape(watch_accessed_variables=False) as tape:
#     prediction = get_start_prediction()
#     word_index = 1
    
#     print(tar.shape, prediction.shape)
#     for target in tar:
#       # print(inp.shape, prediction.shape)
#       pred = model(inputs=[inp, prediction], training=True)
      
#       pred_index = np.argmax(pred[0][word_index].numpy())
#       pred_word = gs.index_to_key[pred_index]
      
#       prediction[word_index] = pred_index
#       word_index += 1
      
#       print(pred_index, pred_word)
      
#       loss = loss_function(target, pred[0][word_index])
#       tape.watch(loss)
      
#       if(get_pad_index(target)):
#         print("return")
#         break
    
#     # print(target, " : ", prediction)
      
#     gradients = tape.gradient(loss, model.trainable_variables)
      
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      
#     train_loss(loss)
#     train_accuracy(accuracy_function(tar, pred))
    
#     # print(loss)
      
#   # print(gradients)
#   # exit()

for epoch in range(epochs):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()
  
  inp, tar = train_dataset
  for i in range(len(inp)):
    # print(inp[i], " : ", tar[i])
    train_step(inp[i], tar[i])

#   if (epoch + 1) % 5 == 0:
#     ckpt_save_path = ckpt_manager.save()
#     print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
  
  if(epoch == 1):
    model.save('model1')
  else:
    model.save('model2')
        
            