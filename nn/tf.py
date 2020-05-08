import tensorflow as tf
from tensorflow.python import placeholder, global_variables_initializer
from tensorflow.python.platform.app import run
from tensorflow.python.training.adam import AdamOptimizer

data = tf.keras.datasets.mnist.load_data()
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = placeholder('float', [None, 784])
y = placeholder('float')


def neural_network(data):
    hidden_lalyer_1 = {
        'weights': tf.Variable(tf.random.normal([784, n_nodes_hl1])),
        'biases': tf.Variable(tf.random.normal(n_nodes_hl1))
    }
    hidden_lalyer_2 = {
        'weights': tf.Variable(tf.random.normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random.normal(n_nodes_hl2))
    }

    hidden_lalyer_3 = {
        'weights': tf.Variable(tf.random.normal([n_nodes_hl2, n_nodes_hl3])),
        'biases': tf.Variable(tf.random.normal(n_nodes_hl3))
    }
    output_layer = {
        'weights': tf.Variable(tf.random.normal([n_nodes_hl3, n_classes])),
        'biases': tf.Variable(tf.random.normal(n_classes))
    }

    l1 = tf.add(tf.linalg.matmul(data, hidden_lalyer_1['weights']) + hidden_lalyer_1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.linalg.matmul(data, hidden_lalyer_2['weights']) + hidden_lalyer_2['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.linalg.matmul(data, hidden_lalyer_3['weights']) + hidden_lalyer_3['biases'])
    l3 = tf.nn.relu(l3)

    out = tf.linalg.matmul(data, output_layer['weights']) + output_layer['biases']
    return out


def train_nn(x):
    prediction = neural_network(x)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizier = AdamOptimizer().minimize(cost_func)

    hm_empochs = 10

    global_variables_initializer()


    for epoch in range(hm_empochs):
        epoch_loss = 0
        for _ in range(int(data.train.numexamples/batch_size)):
            x, y = data.train.next_batch(batch_size)
            _, c = run([optimizier, cost_func])

            def runfunc(feed_dict):
                pass



