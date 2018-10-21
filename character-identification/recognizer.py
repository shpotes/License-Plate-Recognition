import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time, os
from util.dataset import DataSet
from datetime import timedelta
import pickle

data = pickle.load(open('../data/letters/dataset.pkl', 'rb' ))

def plot_images(images, cls_true, cls_pred=None):    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i], cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(data.class_names[cls_true[i]])
        else:
            xlabel = "True: {0}, Pred: {1}".format(data.class_names[cls_true[i]], data.class_names[cls_pred[i]])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

x = tf.placeholder(tf.float32, [None, data.img_size_flat], name='x')
y_true = tf.placeholder(tf.float32, [None, data.num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

new_weights = lambda shape: tf.Variable(tf.truncated_normal(shape, stddev=0.05))
new_biases = lambda length: tf.Variable(tf.constant(0.05, shape=[length]))
    
def new_fc_layer(inp, num_inputs, num_outputs, use_relu=True, keep_prob=0):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = inp
    
    if keep_prob:
        layer = tf.nn.dropout(inp, keep_prob) * keep_prob

    layer = tf.matmul(inp, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


fc2_size = 64 

layer_fc2 = new_fc_layer(inp=x, num_inputs=data.img_size_flat, num_outputs=fc2_size, use_relu=True, keep_prob=0.3)
layer_fc3 = new_fc_layer(inp=layer_fc2, num_inputs=fc2_size, num_outputs=data.num_classes)

y_pred = tf.nn.softmax(layer_fc3)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc3, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64

total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    start_time = time.time()

    for i in range(total_iterations, total_iterations + num_iterations):
        x_batch, y_true_batch, y_batch_cls = data.random_batch(batch_size=train_batch_size)

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



optimize(10000)

## Read data
X = []
M = input('new data folder')
os.chdir(M)

for i in os.listdir():
    img = cv2.cvtColor(cv2.resize(cv2.imread(file), self.img_shape), cv2.COLOR_BGR2GRAY)
    X.append(img.flatten())

pred = session.run(y_pred_cls, feed_dict={x: X})
print(pred)
