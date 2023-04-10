import tensorflow as tf
from razdel import tokenize

def create_dataset(data_path, gs, max_length=300):
    # Read the data from the file
    lines = open(data_path, encoding='UTF-8').read().strip().lower().split('\n')
    
    queries = []
    targets = []
    
    for str in lines:
        # print(str)
        query, target = str.split(':')
        print(query, target)
        
        q = []
        for token in tokenize(query):
            idx = 0
            try:
                idx = gs.key_to_index[token.text]
            except:
                print("can't find " + token.text)
                idx = gs.key_to_index['<unk>']
            q.append(idx)
        
        
        # TODO: optimize this
        pad = gs.key_to_index['<pad>']
        while(len(q) < max_length):
            q.append(pad)
        queries.append(q)
        
        t = []
        # add <start>
        t.append(gs.key_to_index['<start>'])
        for token in tokenize(target):
            idx = 0
            try:
                idx = gs.key_to_index[token.text]
            except:
                print("can't find " + token.text)
                idx = gs.key_to_index['<unk>']
            t.append(idx)
        # add <end>
        t.append(gs.key_to_index['<end>'])
            
        while(len(t) < max_length):
            t.append(pad)
            
        targets.append(t)

    
    # Create the dataset
    return (np.array(queries), np.array(targets))


def create_from_promt(query, gs, max_length=300):
    q = []
    for token in tokenize(query):
        idx = 0
        try:
            idx = gs.key_to_index[token.text]
        except:
            print("can't find " + token.text)
            idx = gs.key_to_index['<unk>']
        q.append(idx)
    
    
    # TODO: optimize this
    pad = gs.key_to_index['<pad>']
    while(len(q) < max_length):
        q.append(pad)
        
    return np.array(q)
            
import numpy as np            

def read_from_pred(pred, gs, max_length=300):
    q = []
    for seq in pred[0]:
        idx = np.argmax(seq)
        word = gs.index_to_key[idx]
        # print(idx, word)
        q.append(word)
    return q

def get_pad_index(idx, gs):
    return idx == gs.key_to_index['<pad>']

def get_start_prediction(gs, max_length=300):
    prediction = [gs.key_to_index['<start>']]
    
    # TODO: optimize this
    pad = gs.key_to_index['<pad>']
    while(len(prediction) < max_length):
        prediction.append(pad)
        
    return np.array(prediction)