import numpy as np

max_word_len = 128

class NavecModel:
    
    def __init__(self):
        from navec import Navec

        path_model = 'navec_hudlit_v1_12B_500K_300d_100q.tar'

        navec = Navec.load(path_model)
        self.embedding_dim = 300

        self.gs = navec.as_gensim

        self.gs.add_vector('<start>', np.full((self.embedding_dim,), -1))
        self.gs.add_vector('<end>', np.full((self.embedding_dim,), 1))
        
        self.pad = self.gs.key_to_index['<pad>']
        self.start = self.gs.key_to_index['<start>']
        self.end = self.gs.key_to_index['<end>']
        self.unk = self.gs.key_to_index['<unk>']
        
        self.navec = navec
        self.vocab_size = self.gs.vectors.shape[0]
        
        
    def get_weights(self):
        return self.gs.vectors
        
    def get_embedding_dim(self):
        return self.embedding_dim
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_gensim_model(self):
        return self.gs
    
    def create_dataset_text(self, data_path):
        from razdel import tokenize
        from collections import Counter
        
        # Read the data from the file
        lines = open(data_path, encoding='UTF-8').read().strip().lower().split('\n')
        
        queries = []
        targets = []
        
        line_index = 0
        lines_count = len(lines)
        
        for str in lines:
            words = str.split()
            for i in range(0, len(words), max_word_len-2):
                query = ' '.join(words[i:i+max_word_len-2])
                # print(query)
                q = []
                t = []
                q.append(self.gs.key_to_index['<start>'])
                idxCount = 1
                
                for token in tokenize(query):
                    try:
                        if(idxCount >= 62):
                            break
                        idx = int(self.gs.key_to_index[token.text])
                        q.append(idx)
                        t.append(idx)
                        # print(idx)
                        idxCount += 1
                    except:
                        # print("can't find " + token.text)
                        # idx = gs.key_to_index['<unk>']
                        pass
                    
                t.append(self.gs.key_to_index['<end>'])
                targets.append(t)
                queries.append(q)
            
            line_index += 1
            print("Dataset loading: %.0f%s" % (((line_index / lines_count) * 100), '%'))
            
        # Create the dataset
        return (queries, targets)                

    def create_dataset(self, data_path):
        from razdel import tokenize
        
        # Read the data from the file
        lines = open(data_path, encoding='UTF-8').read().strip().lower().split('\n')

        queries = []
        targets = []

        line_index = 0
        lines_count = len(lines)

        for str in lines:
            # print(str)
            query, target = str.split('[split]')

            q = []
            q.append(self.gs.key_to_index['<start>'])
            for token in tokenize(query):
                try:
                    idx = self.gs.key_to_index[token.text]
                    q.append(idx)
                except:
                    print("can't find " + token.text)
                    # idx = gs.key_to_index['<unk>']
                    pass
                
            # q.append(self.gs.key_to_index['<end>'])
            queries.append(q)

            t = []
            # add <start>
            # t.append(self.gs.key_to_index['<start>'])
            for token in tokenize(target):
                # idx = 0
                try:
                    idx = self.gs.key_to_index[token.text]
                    t.append(idx)
                except:
                    print("can't find " + token.text)
                    # idx = gs.key_to_index['<unk>']
                    pass
                
            # add <end>
            t.append(self.gs.key_to_index['<end>'])

            targets.append(t)
            line_index += 1
            print("Dataset loading: %.0f%s" % (((line_index / lines_count) * 100), '%'))

        # Create the dataset
        return (queries, targets)


    def create_from_prompt(self, query):
        from razdel import tokenize
        q = []
        for token in tokenize(query):
            idx = 0
            try:
                idx = self.gs.key_to_index[token.text]
                q.append(idx)
            except:
                pass
                # print("can't find " + token.text)
                # idx = gs.key_to_index['<unk>']

        return q
            
    def read_from_pred(self, pred):
        q = []
        for seq in pred[0]:
            idx = np.argmax(seq)
            word = self.gs.index_to_key[idx]
            # print(idx, word)
            q.append((word, idx))
        return q

    def is_end_index(self, idx):
        return idx == self.end

    def is_pad_index(self, idx):
        return idx == self.pad

    def get_pad_index(self):
        return self.pad

    def get_pad_vector(self):
        return self.gs.get_vector('<pad>')

    def is_unk_index(self, idx):
        return idx == self.unk
    
    def get_start_prediction(self):
        prediction = [self.start]
        
        return prediction


class TransformerModel:

    def __init__(self, navec: NavecModel, max_word_len = 150, 
                     num_heads = 15,
                     dff = 16384,
                     dropout_rate = 0.1,
                     mha_dropout_rate = 0.1,
                     learning_rate = 0.001,
                     num_layers = 16):

        import tensorflow as tf
        from tensorflow import range, int64, int32
        from keras.layers import Input, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention, Concatenate
        from keras.optimizers import Adam
        from keras.models import Model
        from utilities import PositionalEncoding, MaskedSelfAttention, CustomSchedule
        
        self.navec = navec
        self.max_word_len = max_word_len
        self.embedding_dim = navec.get_embedding_dim()
        self.vocab_size = navec.get_vocab_size()
        self.word_pos = 0

        with tf.device('/GPU'):

            # Define input layer
            encoder_inputs = Input(shape=(max_word_len,), dtype=int32)
            decoder_inputs = Input(shape=(max_word_len,), dtype=int32)

            weights = navec.get_weights()

            # Add embedding layer
            embed = Embedding(input_dim=weights.shape[0], 
                              output_dim=weights.shape[1], 
                              weights=[weights],
                              trainable=False)
            embed_input = embed(encoder_inputs)
            embed_output = embed(decoder_inputs)

            # Add positional encoding to embedding
            input_embedded = PositionalEncoding(self.max_word_len, self.embedding_dim)(embed_input)
            output_embedded = PositionalEncoding(self.max_word_len, self.embedding_dim)(embed_output)

            # Add encoder layers
            encoder_output = input_embedded

            # Define Feed Forward NN
            feed_forward_input = None
            feed_forward = None
            feed_forward_output = None

            # Add encoder layers
            for i in range(num_layers):
                # Add multi head Attention Layer
                attn = MultiHeadAttention(self.embedding_dim, num_heads, dropout=mha_dropout_rate)(encoder_output, encoder_output)
                attn = Dropout(dropout_rate)(attn)
                attn = LayerNormalization(epsilon=1e-6)(encoder_output + attn)

                # Add Feed Forward NN
                if(feed_forward_input is None):
                  feed_forward_input = Dense(dff, activation='relu')(attn)
                  feed_forward = Dense(self.embedding_dim)(feed_forward_input)
                  feed_forward_output = Dropout(dropout_rate)(feed_forward)
                else:
                  Concatenate([attn, feed_forward_input])

                encoder_output = LayerNormalization(epsilon=1e-6)(attn + feed_forward_output)


            # Add decoder layers
            decoder_output = output_embedded
            for i in range(num_layers):
                # Adding Masked self attention layer
                masked_self_attn = MaskedSelfAttention(self.embedding_dim, num_heads, dropout_rate=mha_dropout_rate)(decoder_output, decoder_output)
                masked_self_attn = LayerNormalization(epsilon=1e-6)(decoder_output + masked_self_attn)

                # Add multi head Attention Layer
                attn = MultiHeadAttention(self.embedding_dim, num_heads, dropout=mha_dropout_rate)(masked_self_attn, encoder_output)
                attn = Dropout(dropout_rate)(attn)
                attn = LayerNormalization(epsilon=1e-6)(masked_self_attn + attn)

                # Add Feed Forward NN

                Concatenate([attn, feed_forward_input])

                decoder_output = LayerNormalization(epsilon=1e-6)(attn + feed_forward_output)


            # Add output layer
            outputs = Dense(units=self.vocab_size, activation='softmax')(decoder_output)

            # Create model
            self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

            # Define loss function and optimizerW
            
            if(learning_rate is None):
                self.learning_rate = CustomSchedule(self.embedding_dim)
            else:
                self.learning_rate = learning_rate
            self.optimizer = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

            checkpoint_path = "./checkpoints/train4"

            self.ckpt = tf.train.Checkpoint(transformer=self.model,
                                     optimizer=self.optimizer)

            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=5)
            # if a checkpoint exists, restore the latest checkpoint.
            if self.ckpt_manager.latest_checkpoint:
              self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
              print('Latest checkpoint restored!!')

            
   
    def summary(self):
        return self.model.summary()
    
    def predict(self, inp, input_pred):
        # print(inp)
        # print(input_pred)
        res = self.model(inputs=[inp, input_pred], training=False)
        
        word, idx = self.navec.read_from_pred(res)[self.word_pos]
        self.word_pos += 1
        
        return (word, idx, self.word_pos)
    
    def prepare_for_pred(self, prompt):
        prompt = self.navec.create_from_prompt(prompt.lower())

        prediction = self.navec.get_start_prediction()

        inp = np.array(prompt)
        inp = np.append(inp, np.full(self.max_word_len - (len(inp)), self.navec.get_pad_index()))

        input_pred = np.array(prediction)
        input_pred = np.append(input_pred, np.full(self.max_word_len - (len(input_pred)), self.navec.get_pad_index()))
        
        self.word_pos = 0
        
        return (inp, input_pred)
    
    
    



class OpenAiTransformerModel:

    def __init__(self, navec: NavecModel, max_word_len = 150, 
                     num_heads = 15,
                     dff = 16384,
                     dropout_rate = 0.1,
                     mha_dropout_rate = 0.1,
                     learning_rate = 0.001,
                     num_layers = 16):

        import tensorflow as tf
        from tensorflow import range, int64
        from keras.layers import Input, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention, Concatenate
        from keras.optimizers import Adam
        from keras.models import Model
        from utilities import PositionalEncoding, MaskedSelfAttention, CustomSchedule
        
        self.navec = navec
        self.max_word_len = max_word_len
        self.embedding_dim = navec.get_embedding_dim()
        self.vocab_size = navec.get_vocab_size()
        self.word_pos = 0

        with tf.device('/GPU'):

            # Define input layer
            encoder_inputs = Input(shape=(max_word_len,), dtype=int64)
            decoder_inputs = Input(shape=(max_word_len,), dtype=int64)

            weights = navec.get_weights()

            # Add embedding layer
            embed = Embedding(input_dim=weights.shape[0], 
                              output_dim=weights.shape[1], 
                              weights=[weights],
                              trainable=False)
            embed_input = embed(encoder_inputs)
            embed_output = embed(decoder_inputs)

            # Add positional encoding to embedding
            input_embedded = PositionalEncoding(self.max_word_len, self.embedding_dim)(embed_input)
            output_embedded = PositionalEncoding(self.max_word_len, self.embedding_dim)(embed_output)

            # Add encoder layers
            encoder_output = input_embedded

            # Define Feed Forward NN
            feed_forward_input = None
            feed_forward = None
            feed_forward_output = None

            # Add decoder layers
            decoder_output = output_embedded
            for i in range(num_layers):
                # Adding Masked self attention layer
                masked_self_attn = MaskedSelfAttention(self.embedding_dim, num_heads, dropout_rate=mha_dropout_rate)(decoder_output, decoder_output)
                masked_self_attn = LayerNormalization(epsilon=1e-6)(decoder_output + masked_self_attn)

                # Add multi head Attention Layer
                attn = MultiHeadAttention(self.embedding_dim, num_heads, dropout=mha_dropout_rate)(masked_self_attn, encoder_output)
                attn = Dropout(dropout_rate)(attn)
                attn = LayerNormalization(epsilon=1e-6)(masked_self_attn + attn)

                # Add Feed Forward NN
                if(feed_forward_input is None):
                  feed_forward_input = Dense(dff, activation='relu')(attn)
                  feed_forward = Dense(self.embedding_dim)(feed_forward_input)
                  feed_forward_output = Dropout(dropout_rate)(feed_forward)
                else:
                  Concatenate([attn, feed_forward_input])

                decoder_output = LayerNormalization(epsilon=1e-6)(attn + feed_forward_output)


            # Add output layer
            outputs = Dense(units=self.vocab_size, activation='softmax')(decoder_output)
            # outputs = Dense(units=self.vocab_size)(decoder_output)

            # Create model
            self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

            # Define loss function and optimizerW
            
            if(learning_rate is None):
                self.learning_rate = CustomSchedule(self.embedding_dim)
            else:
                self.learning_rate = learning_rate
            self.optimizer = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

            checkpoint_path = "./checkpoints/train"

            self.ckpt = tf.train.Checkpoint(transformer=self.model,
                                     optimizer=self.optimizer)

            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=5)
            # if a checkpoint exists, restore the latest checkpoint.
            if self.ckpt_manager.latest_checkpoint:
              self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
              print('Latest checkpoint restored!!')

            
   
    def summary(self):
        return self.model.summary()
    
    def predict(self, inp, input_pred):
        # print(inp)
        # print(input_pred)
        res = self.model(inputs=[inp, input_pred], training=False)
        
        word, idx = self.navec.read_from_pred(res)[self.word_pos]
        self.word_pos += 1
        
        return (word, idx, self.word_pos)
    
    def prepare_for_pred(self, prompt):
        prompt = self.navec.create_from_prompt(prompt.lower())

        prediction = self.navec.get_start_prediction()

        inp = np.array(prompt)
        inp = np.append(inp, np.full(self.max_word_len - (len(inp)), self.navec.get_pad_index()))

        input_pred = np.array(prediction)
        input_pred = np.append(input_pred, np.full(self.max_word_len - (len(input_pred)), self.navec.get_pad_index()))
        
        self.word_pos = 0
        
        return (inp, input_pred)