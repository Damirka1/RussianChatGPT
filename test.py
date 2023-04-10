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

from utilities import PositionalEncoding, MaskedSelfAttention, create_masks

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
print(gs.key_to_index['привет'],  gs.index_to_key[335377])

# add to gensim tag words
gs.add_vector('<start>', np.full((300,), -1))
gs.add_vector('<end>', np.full((300,), 1))

# print tags as vectors
print(gs.get_vector('<start>'))
print(gs.get_vector('<end>'))

# Define hyperparameters
max_word_len = 300
embedding_dim = 300
num_heads = 4
dff = 512
dropout_rate = 0.5
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

# Compile model
# model.compile()

print(model.output_shape)
# print(model.summary())

# exit()

# training

from dataset import create_dataset, get_pad_index

train_dataset = create_dataset('train.txt', gs, max_word_len);

# # Define loss function and optimizer
# loss_object = SparseCategoricalCrossentropy(from_logits=True)
optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# # Define metrics to track during training
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


# def loss_function(real, pred):
#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     loss_ = loss_object(real, pred)
    
#     mask = tf.cast(mask, dtype=loss_.dtype)
#     loss_ *= mask
    
#     return tf.reduce_mean(loss_)

def get_start_prediction():
    prediction = [gs.key_to_index['<start>']]
    
    # TODO: optimize this
    pad = gs.key_to_index['<pad>']
    while(len(prediction) < max_word_len):
        prediction.append(pad)
        
    return np.array(prediction)

# print(get_start_prediction())

# for epoch in range(epochs):
#     queries, targets = train_dataset
#     for i in range(len(queries)):
#         query = np.array(queries[i])
#         target = np.array(targets[i])
        
#         # print(query.shape)
#         # exit()
    
#         # TODO: need to convert splited targets by words to softmax output    
#         for i in range(epochs):
#             with tf.GradientTape() as tape:
#                 prediction = get_start_prediction()
#                 word_index = 1
#                 pred = model(inputs=[query, prediction], training=True)
                    
#                 # pred_index = np.argmax(pred[0][0])
#                 # pred_word = gs.index_to_key[pred_index]
                
#                 # prediction[word_index] = pred_index
#                 # word_index += 1
                
#                 # print(epoch, target_word, pred_index, pred_word)
                
#                 loss = loss_function(target, pred[0])
#                 tape.watch(loss)
#                 print(loss)
                    
#             # Compute the gradients and update the weights
#             # print(model.trainable_variables)
#             print(loss)
#             gradients = tape.gradient(loss, model.trainable_variables)
            
#             # Print the gradients
#             for var, grad in zip(model.trainable_variables, gradients):
#                 print(var.name, grad)
            
#             exit()
#             optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        
#             # Update the training metrics
#             train_loss(loss)
#             train_accuracy(target, prediction)
                
                 
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

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

train_step_signature = [
    tf.TensorSpec(shape=(max_word_len,), dtype=tf.int64),
    tf.TensorSpec(shape=(max_word_len,), dtype=tf.int64),
]

# def get_predict(pred, word_index):
  
#   print(pred_index, pred_word)
  

# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar):

  
  with tf.GradientTape() as tape:
    prediction = get_start_prediction()
    word_index = 1
    
    for target in tar:
      # print(inp.shape, prediction.shape)
      pred = model(inputs=[inp, prediction], training=True)
      
      pred_index = np.argmax(pred[0][word_index].numpy())
      pred_word = gs.index_to_key[pred_index]
      
      prediction[word_index] = pred_index
      word_index += 1
      
      # print(pred_index, pred_word)
      
      # print(np.expand_dims(target, axis=0).shape, np.expand_dims(prediction, axis=0).shape)
      
      loss = loss_function(target, pred[0][word_index])
      
      if(get_pad_index(target, gs)):
        break
      
    # print(loss)
      
    gradients = tape.gradient(loss, model.trainable_variables)
      
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      
    train_loss(loss)
    train_accuracy(accuracy_function(tar, pred))
      
  # print(gradients)
  # exit()
  

for epoch in range(epochs):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  for (inp, tar) in train_dataset:
    train_step(inp, tar)

#   if (epoch + 1) % 5 == 0:
#     ckpt_save_path = ckpt_manager.save()
#     print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')                
  
  if(epoch == 10):
    model.save('model')
    exit()
        
            