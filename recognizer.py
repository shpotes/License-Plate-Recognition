import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time, cv2
from util.dataset import DataSet
from util.network import new_fc_layer
from datetime import timedelta
import pickle

# Define hyperparameters
LR = 1e-4
TRAIN_BATCH = 64
DROP = 0.5
FC1_SIZE = 40
NITER = 4000
TRAIN = ""

# Read data
data = pickle.load(open('data.pkl', 'rb'))

x = tf.placeholder(tf.float32, [None, data.img_size_flat], name='x')
y_true = tf.placeholder(tf.float32, [None, data.num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_fc1 = new_fc_layer(inp=x, num_inputs=data.img_size_flat,
                         num_outputs=FC1_SIZE, use_relu=True,
                         keep_prob=DROP)

layer_fc2 = new_fc_layer(inp=layer_fc1, num_inputs=FC1_SIZE,
                         num_outputs=data.num_classes)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc3,
                                                           labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not TRAIN:
    optimize(NITER, optimizer, data, sess)
    print_test_accuracy()
else:
    pass # TODO https://www.tensorflow.org/guide/saved_model

def prediction(plate, data):
    X = []
    for i in plate:
        X.append(255 - cv2.resize(i, data.img_shape).flatten())
        
    X = np.vstack(X)
    return sess.run(y_pred, feed_dict={x:X})

sess.close()
