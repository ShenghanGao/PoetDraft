from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from reader import getDicts
from reader import read_poems
import os

hidden_size = 200
num_steps = 20
batch_size = 20
keep_prob = 0.5
num_layers = 2
vocab_size = 2000
max_grad_norm = 5
lr_decay = 0.5

class trainModel(object):
    def __init__(self, training, infer=False):
        def print_out_w(tensor):
            out = tf.Print(tensor, [tensor], message='softmax_w is ', summarize=20)
            out1 = out
            return out1
        def print_out_b(tensor):
            out = tf.Print(tensor, [tensor], message='softmax_b is ', summarize=20)
            out1 = out
            return out1

        if infer:
            batch_size = 1
            num_steps = 1
        else:
            batch_size = 20
            num_steps = 20

        #define placeholders
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._label = tf.placeholder(tf.int32, [batch_size, num_steps])
    
        #define lstm cells
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)
        if training and keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

        self.cell = cell
    
        #define initial states
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for step in range(num_steps):
                if step > 0: 
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, step, :], state)
                outputs.append(cell_output)
    
        #get output
        output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
    
        #get softmax
        w = tf.get_variable("w", [hidden_size, vocab_size])
        #w = print_out_w(w) 
        b = tf.get_variable("b", [vocab_size])
        #b = print_out_b(b)
        logits = tf.add(tf.matmul(output, w) , b)
    
        #compute loss
        reshaped_label = tf.reshape(self._label, [-1])
        loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [reshaped_label], [tf.ones([batch_size * num_steps])])

        #compute cost
        cost = tf.reduce_sum(loss) / batch_size
        self._cost = cost
        self._final_state = state

        #add probabilities and store logits
        self._probabilities = tf.nn.softmax(logits)
        self._logits = logits

        if not training:
            return 

        #set learning rate
        self._learning_rate = tf.Variable(0.0, trainable=False)
        train_vars = tf.trainable_variables()

        #get gradients    
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, train_vars), max_grad_norm)
    
        #optimize trainble variables
        optimizer = tf.train.MomentumOptimizer(self._learning_rate, 0.5)
        zipped_value = zip(grads, train_vars)
        self._train = optimizer.apply_gradients(zipped_value)

        self._saver = tf.train.Saver(tf.all_variables())


    def sample(self, session, index_to_char, char_to_index, prime, num=40):

        #print('aaaaaaaaa', type(prime), 'bbbbbb', len(prime))
        prime = '^' + prime.decode('utf-8')
        print('aaaaaaaaa', type(prime), 'bbbbbb', len(prime), 'ccccccc  ', prime)

        #for key in char_to_index:
            #print(key, ': ', char_to_index[key])

        state = self.cell.zero_state(1, tf.float32).eval()

        #prime = prime.encode('utf-8')

        for char in prime:
            #char = char.decode('utf-8')
            x = np.zeros((1, 1))
            x[0, 0] = char_to_index[char]

            #print(char)


            feed = {self._input_data: x, self._initial_state: state}
            [state] = session.run([self._final_state], feed)

        poem = prime
        char = prime[-1]

        for n in range(num):
            x = np.zeros((1, 1))
            feed = {self._input_data: x, self._initial_state: state}
            [probs, state] = session.run([self._probabilities, self._final_state], feed)
            p = probs[0]

            #print('p shape ', tf.shape(p))
            #print('p sum ', np.sum(p))
            #print('p ', p)

            sample = np.argmax(p)

            pred = index_to_char[sample]

            #print(pred.encode('utf8'))

            #exit()

            char = pred
            poem += pred

        return poem

def train(session, train_model, data, eval_op, index_to_char, verbose=False):
    #epoch_size = ((len(data) // batch_size) - 1) // num_steps
    cost_sum = 0.0
    iteration_num = 0
    state = train_model._initial_state.eval()
    step = 0
        
    
    #print("data shape ", data.shape[0], data.shape[1])
    
    #get batches from data
    for i in range(20):
        for j in range(data.shape[1] - num_steps):
            x = data[i * batch_size : (i + 1) * batch_size, j : j +  num_steps]
            y = data[i * batch_size : (i + 1) * batch_size, j + 1 : j + 1 + num_steps]
           
            #print out x y shape
            #print("x shape ", x.shape)
            #print("y shape ", y.shape)
 
            cost, state, probs, logits, _ = session.run([train_model._cost, train_model._final_state, train_model._probabilities, train_model._logits, eval_op], {train_model._input_data: x, train_model._label: y, train_model._initial_state: state})
            cost_sum += cost
            iteration_num += num_steps
            step += 1
            #print("probs shape ", probs.shape)
            #print(probs[0])
            if verbose:
                print("t perplexity: %.3f" % (np.exp(cost_sum / iteration_num)))
            else:
                #print("v perplexity: %.3f" % (np.exp(cost_sum / iteration_num)))
 
                chosen_word = np.argmax(probs, 1)
                
                index = chosen_word[-1]
                word = index_to_char[index]
                print("prediction word ", word)

    return np.exp(cost_sum / iteration_num)


def main(_):

    #load data
    index_to_char, char_to_index = getDicts(vocab_size)
    data = read_poems(char_to_index)
    
    #print(data.shape)
    train_data = data[1000:147541]
    val_data = data[0:1000]

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("trainModel", reuse=None, initializer=initializer):
            t_train = trainModel(training=True)
        with tf.variable_scope("trainModel", reuse=True, initializer=initializer):
            t_valid = trainModel(training=False)
  
        tf.initialize_all_variables().run()

        print("-------", type(tf.all_variables()))

        for a in tf.all_variables():
            print(a.name)

        print("-------")

        for i in range(1):
            #let learning rate decay 
            learning_decay = lr_decay ** max(i, 0.0)
            tf.assign(t_train._learning_rate, learning_decay).eval()
            print("Epoch ", i+1)
            #train
            train_perplexity = train(session, t_train, train_data, t_train._train, index_to_char, verbose=True)
            print("train_perplexity: ", train_perplexity)
            checkpoint_path = os.path.join("", 'model.ckpt')
            t_train._saver.save(session, checkpoint_path, global_step=i)
            #print("have saved checkpoint")
            #validate
            val_perplexity = train(session, t_valid, val_data, tf.no_op(), index_to_char)
            print("val_perplexity: ", val_perplexity)


if __name__ == "__main__":
    tf.app.run() 






