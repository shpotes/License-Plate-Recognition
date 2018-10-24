import tensorflow as tf

def new_fc_layer(inp, num_inputs, num_outputs, use_relu=True, keep_prob=0):
    weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs],
                                              stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
    layer = inp

    if keep_prob:
        layer = tf.nn.dropout(inp, keep_prob) * 1/keep_prob

    layer = tf.matmul(inp, weights) + biases

    if use_relu:
        layer = tf.nn.elu(layer) # Ja!

    return layer

def optimize(num_iterations, optimizer, data, session):
    start_time = time.time()
    for i in range(num_iterations):
        x_batch, y_true_batch, _ = data.random_batch(batch_size=TRAIN_BATCH)  
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            print("Iteration: {0:>6}, Accuracy: {1:>6.1%}".format(i, acc))

    end_time = time.time()
    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def print_test_accuracy(test_batch_size=256):
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

