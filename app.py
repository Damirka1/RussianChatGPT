from model import *
import tensorflow as tf

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_virtual_device_configuration(gpus[0], 
   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3550)])


# Define hyperparameters
# max_word_len = 150
# embedding_dim = 300
# num_heads = 15
# dff = 16384
# dropout_rate = 0.1
# mha_dropout_rate = 0.1
# learning_rate = 0.001
epochs = 0
# num_layers = 16

navec_model = NavecModel()

transformer_model = TransformerModel(navec=navec_model, learning_rate=None)

# print(transformer_model.summary())

# train_dataset = navec_model.create_dataset('df.txt');
# train_dataset = navec_model.create_dataset('train.txt');

# transformer_model.train(epochs, train_dataset)

text = input("Введите запрос: ")

while text != "exit":
  inp, input_pred = transformer_model.prepare_for_pred(text)

  print("Мой ответ: ", end="")
  
  while True:
    word, idx, word_pos = transformer_model.predict(inp, input_pred)
    
    if(navec_model.is_end_index(idx)):
        break
      
    input_pred[word_pos] = idx
    print(word, " ", sep="", end="", flush=True)
    
  print()
  
  text = input("Введите запрос: ")
    
        
            