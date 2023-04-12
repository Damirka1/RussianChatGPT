import tensorflow as tf
from razdel import tokenize

pad = 0
unk = 0
end = 0

def create_dataset(data_path, gs, max_length=300):
    # Read the data from the file
    lines = open(data_path, encoding='UTF-8').read().strip().lower().split('\n')
    
    queries = []
    targets = []
    
    global pad
    global end
    global unk
    pad = gs.key_to_index['<pad>']
    end = gs.key_to_index['<end>']
    unk = gs.key_to_index['<unk>']
    
    line_index = 0
    lines_count = len(lines)
    
    for str in lines:
        # print(str)
        query, target = str.split('[split]')
        # print(query, target)
        
        q = []
        for token in tokenize(query):
            # idx = 0
            try:
                idx = gs.key_to_index[token.text]
                q.append(idx)
            except:
                # print("can't find " + token.text)
                # idx = gs.key_to_index['<unk>']
                pass
        
        q = q[:max_length]
        
        
        # TODO: optimize this
        
        while(len(q) < max_length):
            q.append(pad)
        queries.append(q)
        
        t = []
        # add <start>
        t.append(gs.key_to_index['<start>'])
        for token in tokenize(target):
            # idx = 0
            try:
                idx = gs.key_to_index[token.text]
                t.append(idx)
            except:
                # print("can't find " + token.text)
                # idx = gs.key_to_index['<unk>']
                pass
            
        # add <end>
        if(len(t) >= max_length):
            t = t[:max_length - 1]
        t.append(gs.key_to_index['<end>'])
            
        while(len(t) < max_length):
            t.append(pad)
            
        targets.append(t)
        line_index += 1
        print("Dataset loading: %.0f%s" % (((line_index / lines_count) * 100), '%'))

    
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

def is_end_index(idx):
    global end
    return idx == end

def is_pad_index(idx):
    global pad
    return idx == pad

def is_unk_index(idx):
    global unk
    return idx == unk

def get_start_prediction(gs, max_length=300):
    prediction = [gs.key_to_index['<start>']]
    
    # TODO: optimize this
    pad = gs.key_to_index['<pad>']
    while(len(prediction) < max_length):
        prediction.append(pad)
        
    return np.array(prediction)