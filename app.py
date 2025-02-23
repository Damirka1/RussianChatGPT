from model import *
import tensorflow as tf

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_virtual_device_configuration(gpus[0], 
   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3550)])

# Define hyperparameters OLD MODEL (checkpoint/train4)
# max_word_len = 64
# # embedding_dim = 300
# num_heads = 12
# dff = 512
# dropout_rate = 0.1
# mha_dropout_rate = 0.1
# # learning_rate = None
# learning_rate = 0.01
# epochs = 10000
# num_layers = 12

# Define hyperparameters
max_word_len = 128
# embedding_dim = 300
num_heads = 25
dff = 1024
dropout_rate = 0.2
mha_dropout_rate = 0.1
# learning_rate = None
# learning_rate = 0.0000001
# learning_rate = 0.001
learning_rate = 3e-4
epochs = 10000
num_layers = 24

navec_model = NavecModel()

transformer_model = OpenAiTransformerModel(navec=navec_model,
                                    max_word_len=max_word_len,
                                    num_heads=num_heads,
                                    dff=dff,
                                    num_layers=num_layers,
                                    learning_rate=learning_rate, 
                                    dropout_rate=dropout_rate, 
                                    mha_dropout_rate=mha_dropout_rate)

print(transformer_model.summary())
# exit()

#train_dataset = navec_model.create_dataset_text('df.txt')
train_dataset = navec_model.create_dataset('train.txt')

import train as tn

tn.train(transformer_model, epochs, train_dataset)

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
    
        
            