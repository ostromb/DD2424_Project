import numpy as np
import copy
import pandas as pd
import tensorflow as tf



def load_data(filename,remove_footnotes=False):
    """ Load all characters from text file"""

    with open(filename,encoding='cp850',mode='r') as f:
        data = [c for c in f.read()]
    if remove_footnotes:
        unwanted_chars = ["[","]","(",")","{","}","*","|","<",">","=","#","-","_","^","~","\\","/",":",";","&","@","%","$"]
        for c in unwanted_chars:
            data = [x for x in data if x != c]
    return data

def one_hot_encoding(data, char_to_ind, k):
    """ One hot encoding of data"""
    one_hot = np.zeros((k, len(data)))
    for i, c in enumerate(data):
        one_hot[char_to_ind[c], i] = 1
    return one_hot

def one_hot_decoding(one_hot, ind_to_char):
    """ Decode one hot encoding"""
    data = ''
    for i in range(one_hot.shape[1]):
        data += ind_to_char[np.argmax(one_hot[:, i])]
    return data

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def get_n_grams(text, n):
    """Divide data into n-grams"""
    unwanted_chars =  ["0","1","2","3","4","5","6","7","8","9","[","]","(",")","{","}","*","|","<",">","=","#","-","_","^","~","\\","/",":",";","&","@","%","$"]
    text_cleaned = ""
    for c in text:
        if c not in unwanted_chars:
            text_cleaned += c
    word_list = text_cleaned.split()    
    N = len(word_list)
    data = [""]*(N-n+1)
    for i in range(N-n+1):
        gram = " ".join(word_list[i:i+n])
        data[i] = gram
    return data


def measure_diversity(text_generated, n_max=4):
    """Measures the amount of repetition in a longer generated text, using self-BLUE metric"""
    all_sentences = [s for s in text_generated.split(".") if s]
    N = len(all_sentences)
    bleu_scores = [0]*N
    score = 0
    if N>1:      # Self-bleu does not work for texts with only one sentence
        for i in range(N):
            s = all_sentences[i]
            sentences_copy = copy.deepcopy(all_sentences)
            sentences_copy.remove(s)
            try:
                _, bleu = measure_bleu(s, ".".join(sentences_copy), n_max)
            except:
                print(s)
                print(sentences_copy)
                raise
            bleu_scores[i] = bleu    
    score = np.mean(bleu_scores)    
    return score


def measure_bleu(text_generated, text_val, n_max=4):
    """Measures the fraction of corrrectly spelled words and BLEU score"""
    precision_score = 1
    for n in range(n_max,0, -1):
        words_gen = get_n_grams(copy.deepcopy(text_generated), n)
        words_val = get_n_grams(copy.deepcopy(text_val), n)
        # If there are n-grams of size n in the generated text. Need to check this in case of very short sentences.
        if len(words_gen)>0:
            correct_grams = 0
            output_length = len(words_gen)
            reference_length = len(words_val)
            for gram_gen in words_gen:
                if gram_gen in words_val:
                    correct_grams += 1 
            
            precision = correct_grams/output_length
            precision_score *= precision**(1/n_max) 
    
    fraction_correct_words = precision     # since last iteration of for-loop is 1-grams   
    bleu = precision_score * min(1,output_length/reference_length)     
    return fraction_correct_words, bleu


def split_input_target(batch):
    input_text = batch[:-1]
    """ offset target by one time step """
    target_text = batch[1:]
    return input_text, target_text



def nucleus_sample(predictions_logits, p):
    predictions = tf.nn.softmax(predictions_logits)[0].numpy()
    prob_sum = 0
    largest_probabilities = np.zeros(predictions.shape)
    while prob_sum<p:
        max_ind = np.argmax(predictions)
        max_prob = predictions[max_ind]
        prob_sum += max_prob
        largest_probabilities[max_ind] = max_prob
        predictions[max_ind] = 0
   
    l = [i for i in range(largest_probabilities.shape[-1])]
    probabilities_scaled = copy.deepcopy(largest_probabilities)/prob_sum
    sampled_elem = np.random.choice(l, p=probabilities_scaled)
    return sampled_elem


def generate_text(model, start_string, text_size, char_to_ind, ind_to_char, temp=1.0, p=None):
    # Convert start string to numbers
    input_indices = tf.expand_dims([char_to_ind[s] for s in start_string], 0)

    generated_text = ""
    model.reset_states()
    for i in range(text_size):
        predictions = model(input_indices)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        if p:
            sampled_id = nucleus_sample(predictions, p)
        else:
            # scale probabilities by a temperature to generate more or less predictable text
            predictions = predictions / temp
            # Sample a new character based on the log probability distribution in 'predictions'
            sampled_id = tf.random.categorical(
            predictions,
            num_samples=1
            )[-1,0].numpy()

        # Use sampled char as input for next iteration
        input_indices = tf.expand_dims([sampled_id], 0)
        generated_text += ind_to_char[sampled_id]

    return start_string + generated_text



class RNN:
    def __init__(self, m, k, eta, seq_length, sig):
        self.m = m
        self.k = k
        self.eta = eta
        self.seq_length = seq_length
        self.sig = sig
        self.b = np.zeros((m, 1))
        self.c = np.zeros((k, 1))
        self.U = np.random.normal(0, sig, (m, k))
        self.W = np.random.normal(0, sig, (m, m))
        self.V = np.random.normal(0, sig, (k, m))
        self.hprev = np.zeros((m, 1))
        self.mU = np.zeros_like(self.U)
        self.mV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.mc = np.zeros_like(self.c)
        
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def synthetize(self, h0, x0, n):
        """ Synthetize text"""
        x = x0
        h = h0
        y = np.zeros((self.k, n))

        for t in range(n):
            a = self.W.dot(h)+self.U.dot(x)+self.b
            h = np.tanh(a)
            o = self.V.dot(h)+self.c
            p = self.softmax(o)
        
            cp = np.cumsum(p, axis=0)
            z = np.random.rand()
            ixs = np.nonzero(cp-z > 0)[0]
         
            ii = ixs[0]
            x = np.zeros((self.k, 1))
            x[ii] = 1
            y[:, t] = x[:, 0]
        return y
    
    def forward(self, h0, X, Y):
        """ Forward pass"""
        a = []
        h = []
        o = []
        p = []
        h.append(h0)
        loss = 0

        for t in range(self.seq_length):
            a.append(self.W.dot(h[t])+self.U.dot(X[:, t].reshape(self.k, 1))+self.b)
            h.append(np.tanh(a[t]))
            o.append(self.V.dot(h[t+1])+self.c)
            p.append(self.softmax(o[t]))
            loss += -np.log(np.dot(Y[:, t].reshape(1, self.k), p[t]))[0]
        
        loss = loss/self.seq_length
        
        self.hprev = h[-1]
        return loss, p, h, a, o

    def backward(self, X, Y, p, h, a, o):
        """ Backward pass"""
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
        dc = np.zeros(self.c.shape)
        dh_next = np.zeros((self.m, 1))

        for t in reversed(range(self.seq_length)):
            do = p[t] - Y[:, t].reshape(self.k, 1)
            dV += do.dot(h[t+1].T)
            dc += do
            dh = self.V.T.dot(do) + dh_next
            da = dh * (1 - h[t+1]**2)
            dU += da.dot(X[:, t].reshape(1, self.k))
            db += da
            dW += da.dot(h[t].T)
            dh_next = self.W.T.dot(da)
        for param in [dU, dV, dW, db, dc]:
            np.clip(param, -5, 5, out=param)
        return dU, dV, dW, db, dc
    
    def adagrad(self, X, Y, h0, i):
        """ Train model with adagrad"""
        loss, p, h, a, o = self.forward(h0, X, Y)
        dU, dV, dW, db, dc = self.backward(X, Y, p, h, a, o)
        gamma = 0.9
        for param, dparam, mem in zip([self.U, self.V, self.W, self.b, self.c],
                                  [dU, dV, dW, db, dc],
                                  [self.mU, self.mV, self.mW, self.mb, self.mc]):
            #mem = gamma*mem + (1-gamma)*dparam*dparam if i>0 else dparam * dparam
            mem += dparam * dparam
            param += -self.eta * dparam / np.sqrt(mem + 1e-8)
        return loss
    
    def computeGradsNum(self, X, Y, h, z=1e-6):
        """ Compute gradients numerically"""
        grad_W = np.zeros(self.W.shape)
        grad_U = np.zeros(self.U.shape)
        grad_V = np.zeros(self.V.shape)
        grad_b = np.zeros(self.b.shape)
        grad_c = np.zeros(self.c.shape)
        c = self.forward(h, X, Y)[0]
        for i in range(self.b.shape[0]):
            self.b[i] += z
            c2 = self.forward(h, X, Y)[0]
            grad_b[i] = (c2-c) / z
            self.b[i] -= z
        for i in range(self.c.shape[0]):
            self.c[i] += z
            c2 = self.forward(h, X, Y)[0]
            grad_c[i] = (c2-c) / z
            self.c[i] -= z
        for i in range(self.U.shape[0]):
            for j in range(self.U.shape[1]):
                self.U[i, j] += z
                c2 = self.forward(h, X, Y)[0]
                grad_U[i, j] = (c2-c) / z
                self.U[i, j] -= z
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
                self.V[i, j] += z
                c2 = self.forward(h, X, Y)[0]
                grad_V[i, j] = (c2-c) / z
                self.V[i, j] -= z
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i, j] += z
                c2 = self.forward(h, X, Y)[0]
                grad_W[i, j] = (c2-c) / z
                self.W[i, j] -= z
        return grad_U, grad_V, grad_W, grad_b, grad_c