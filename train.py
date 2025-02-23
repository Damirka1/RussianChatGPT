import numpy as np
import tensorflow as tf
from model import TransformerModel

max_word_len = 128

train_step_signature = [
    tf.TensorSpec(shape=(max_word_len,), dtype=tf.int32),
    tf.TensorSpec(shape=(max_word_len,), dtype=tf.int32),
    tf.TensorSpec(shape=(max_word_len,), dtype=tf.int32),
]

# Define metrics to track during training
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
optimizer = None
model = None
gs = None

pad_index = None

def __masked_loss_function(label, pred):
    mask = label != pad_index
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

def __masked_accuracy_function(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != pad_index

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)

@tf.function(input_signature=train_step_signature)
def step(inp, tar, prediction):
    with tf.GradientTape() as tape:
        pred = model(inputs=[inp, prediction], training=True)
        loss = __masked_loss_function(tar, pred)
        #with tf.device('/CPU:0'):
            
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(__masked_accuracy_function(tar, pred))
    #with tf.device('/CPU:0'):  
        
    return pred

def __train_step(inp, tar):
    
    if(len(inp) <= 1):
        return
    
    prediction = np.array(inp)
    if(len(prediction) < max_word_len):
        prediction = np.append(prediction, np.full(max_word_len - (len(prediction)), pad_index))
        
    input = np.array(tar[:-1])    
    if(len(input) < max_word_len):
        input = np.append(input, np.full(max_word_len - (len(input)), pad_index, dtype=int))
        
    tar_pred = np.array(tar[1:])
    if(len(tar_pred) < max_word_len):
        tar_pred = np.append(tar_pred, np.full(max_word_len - (len(tar_pred)), pad_index, dtype=int))
        
    # print(input)
    # print(prediction)
    # print(tar_pred)
    # exit()
    
    # print("Input: ", input)
    # print("InputStart: ", inp)
    
    # print("Input: ", sep='', end='')
    # for word in input:
    #   print(gs.index_to_key[word], sep='', end=' ')
      
    # print("\nTrue: ", sep='', end='')
    # for word in tar_pred:
    #     print(gs.index_to_key[word], sep='', end=' ')
    
    pred = step(input, tar_pred, prediction)
    
    print("Pred: ", sep='', end='')
    for words in pred[0]:
      word = np.argmax(words.numpy())
      print(gs.index_to_key[word], sep='', end=' ')
    print()
    print(f'Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    print()
    
def train(transformer, epochs, train_dataset):
    import time
    
    global model
    global optimizer
    model = transformer.model
    optimizer = transformer.optimizer
    
    global pad_index
    pad_index = transformer.navec.get_pad_index()
    
    global gs
    gs = transformer.navec.gs
    
    inp, tar = train_dataset
    print("inp count in 1 epoch: ", len(inp))
    
    start_epoch = 0
    start_pos = 0
    
    for epoch in range(start_epoch, epochs):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        #for i in range(start_pos, 2023):
        for i in range(inp):
            print(f'Current train step {i+1}')
            __train_step(inp[i], tar[i])
            if (i + 1) % 100 == 0:
                ckpt_save_path = transformer.ckpt_manager.save()
                print(f'Saving checkpoint for train step {i+1} at {ckpt_save_path}, current epoch - {epoch+1}')
                
        start_pos = 0
                
        # exit()
        # if (epoch + 1) % 10 == 0:
        #     ckpt_save_path = transformer.ckpt_manager.save()
        #     print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
        ckpt_save_path = transformer.ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
        
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')    