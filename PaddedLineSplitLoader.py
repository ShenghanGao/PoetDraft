import numpy as np
import tensorflow as tf

input_file = 'input.txt'

vocab_size = 8
data = []

char_to_id = {}
id_to_char = {}
num_of_char = {}

id_num = 0
max_line_length = 0

with open(input_file, 'r') as f:
    for line in f:
        line = line.strip().decode('utf-8')
        length = len(line)
        max_line_length = max(max_line_length, length)
        for i in range(length):
            c = line[i].encode('utf-8')
            if c not in num_of_char:
                num_of_char[c] = 1
            else:
                num_of_char[c] += 1

ordered = sorted(num_of_char.iteritems(), key=lambda p: p[1], reverse=True)
ordered = ordered[:vocab_size]

for t in ordered:
    id_num += 1
    char_to_id[t[0]] = id_num
    id_to_char[id_num] = t[0]
    # print t[0], t[1]

id_num += 1
char_to_id['*'] = id_num
id_to_char[id_num] = '*'
id_num += 1
char_to_id['^'] = id_num
id_to_char[id_num] = '^'
id_num += 1
char_to_id['$'] = id_num
id_to_char[id_num] = '$'

"""
for char in char_to_id:
    print 'char_to_id[', char, '] = ', char_to_id[char]

for id_num in id_to_char:
    assert char_to_id[id_to_char[id_num]] == id_num
    print 'id_to_char[', id_num, '] = ', id_to_char[id_num]

for char in num_of_char:
    print 'num_of_char[', char, '] = ', num_of_char[char]
"""

line_cnt = 0

with open(input_file, 'r') as f:
    for line in f:
        line = line.strip().decode('utf-8')
        length = len(line)
        data.append([])
        for i in range(max_line_length + 2):
            if i == 0:
                data[line_cnt].append(char_to_id['^'])
            elif 1 <= i <= length:
                c = line[i - 1].encode('utf-8')
                if c in char_to_id:
                    data[line_cnt].append(char_to_id[c])
                else:
                    data[line_cnt].append(char_to_id['*'])
            else:
                data[line_cnt].append(char_to_id['$'])

        line_cnt += 1

for i in range(line_cnt):
    for j in range(max_line_length + 2):
        print id_to_char[data[i][j]],
    print ''

hidden_size = 128
num_layers = 2
num_steps = 20
batch_size = 10
max_grad_norm = 5

class Model(object):
    def __init__(self):
        self._input_data = tf.placeholder(tf.int32, [])
        self._targets = tf.placeholder(tf.int32, [])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
        inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])

        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])]
        )
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
        optimizer = tf.train.MomentumOptimizer(self._lr, 0.9)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

def run_epoch(session, data):
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()

    num_batches = data.shape[0] / batch_size
    epoch_size = data.shape[1] / num_steps

    for i in num_batches:
        for j in epoch_size:
            x = data[num_batches * i : num_batches * (i + 1), epoch_size * j : epoch_size * (j + 1)]
            y = data[num_batches * i : num_batches * (i + 1), epoch_size * j + 1 : epoch_size * (j + 1) + 1]

            cost, state, _ = session.run([m.cost,  m.final_state, eval_op],
                                         {m.input_data: x,
                                          m.targets: y,
                                          m.initial_state: state})

            costs += cost
            iters += m.num_steps

    return np.exp(costs / iters)


def main(_):
    data = reader.ptb_raw_data(path)
    train_data, valid_data, test_data, _ = raw_data

    with tf.Graph().as_default. tf.Session() as session:
        initializer = tf.random_uniform_initializer()

checkpoint_dir = '.'


def sample_sequence(sess, saver, ch):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print 'No checkpoint file found'
        return

    current_state = []

    #for i in range(num_layers):
    #    h_init = tf.zeros([1, ])

    topprint = 20
    poem_cnt = 0

    for i in range(20 * 2):
        furtherexclusion = []
        current_state = []
        for