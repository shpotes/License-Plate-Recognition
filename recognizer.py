import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time, cv2
from util.dataset import DataSet
from datetime import timedelta
import pickle

LR = 1e-4
BATCH = 64
DROP = 0.5
fc2_size = 40
NITER = 4000


def new_fc_layer(inp, num_inputs, num_outputs, use_relu=True, keep_prob=0):
    new_weights = lambda shape: tf.Variable(tf.truncated_normal(shape,
                                                                stddev=0.05))
    new_biases = lambda length: tf.Variable(tf.constant(0.05, shape=[length]))

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = inp

    if keep_prob:
        layer = tf.nn.dropout(inp, keep_prob) * keep_prob

    layer = tf.matmul(inp, weights) + biases

    if use_relu:
        layer = tf.nn.elu(layer) # Ja!

    return layer


data = pickle.load(open('data.pkl', 'rb'))

x = tf.placeholder(tf.float32, [None, data.img_size_flat], name='x')
y_true = tf.placeholder(tf.float32, [None, data.num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)


layer_fc2 = new_fc_layer(inp=x, num_inputs=data.img_size_flat,
                         num_outputs=fc2_size, use_relu=True, keep_prob=DROP)

layer_fc3 = new_fc_layer(inp=layer_fc2, num_inputs=fc2_size,
                         num_outputs=data.num_classes)

y_pred = tf.nn.softmax(layer_fc3)
y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc3,
                                                           labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = BATCH

total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    start_time = time.time()

    for i in range(total_iterations, total_iterations + num_iterations):
        x_batch, y_true_batch, _ = \
                               data.random_batch(batch_size=train_batch_size)

        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(NITER)

test_batch_size = 256

def print_test_accuracy():
    cls_pred = np.zeros(shape=data.num_test, dtype=np.int)
    i = 0

    while i < data.num_test:
        j = min(i + test_batch_size, data.num_test)

        images = data.x_test_flat[i:j, :]
        labels = data.y_test[i:j, :]

        feed_dict = {x: images,
                     y_true: labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = data.y_test_cls

    correct = cls_true.transpose() == cls_pred

    correct_sum = correct.sum()
    acc = float(correct_sum) / data.num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, data.num_test))

print_test_accuracy()


def prediction(plate):
    X = []
    for i in plate:
        X.append(255 - cv2.resize(i, data.img_shape).flatten())

    X = np.vstack(X)
    return session.run(y_pred, feed_dict={x:X})
