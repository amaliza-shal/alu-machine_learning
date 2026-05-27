#!/usr/bin/env python3
"""
Model
"""
import tensorflow as tf


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    builds, trains, and saves a neural network model in tensorflow using
        Adam optimization, mini-batch gradient descent, learning rate decay,
        and batch normalization:
    Data_train is a tuple containing the training inputs and labels,
        respectively
    Data_valid is a tuple containing the validation inputs and labels,
        respectively
    layers is a list containing the number of nodes in each layer of the
        network
    activations is a list containing the activation functions used for each
        layer of the network
    alpha is the learning rate
    beta1 is the weight for the first moment of Adam
    beta2 is the weight for the second moment of Adam
    epsilon is a small number used to avoid division by zero
    decay_rate is the decay rate for inverse time decay of the learning rate
        (the decay step should be 1)
    batch_size is the number of data points in a mini-batch
    epochs is the number of times the training should pass through the whole
        dataset
    save_path is the path where the model should be saved
    Returns: the path where the model was saved
    """
    create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer
    shuffle_data = __import__('2-shuffle_data').shuffle_data

    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    m, nx = X_train.shape
    classes = Y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    prev = x
    for i in range(len(layers)):
        if i == len(layers) - 1:
            # Last layer: no batch norm, use standard dense
            init = tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG")
            layer = tf.layers.Dense(units=layers[i], kernel_initializer=init,
                                    name='layer')
            Z = layer(prev)
            # Output activation (usually softmax for classification)
            prediction = activations[i](Z)
        else:
            prediction = create_batch_norm_layer(prev, layers[i],
                                                 activations[i])
        prev = prediction

    y_pred = prediction
    tf.add_to_collection('y_pred', y_pred)

    loss = tf.losses.softmax_cross_entropy(y, Z)
    tf.add_to_collection('loss', loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1),
                                               tf.argmax(y_pred, 1)),
                                      tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha_decay = tf.train.inverse_time_decay(alpha, global_step, 1,
                                             decay_rate, staircase=True)
    train_op = tf.train.AdamOptimizer(alpha_decay, beta1, beta2,
                                     epsilon).minimize(loss,
                                                       global_step=global_step)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(epochs + 1):
            train_cost, train_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train}
            )
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid}
            )

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if i < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for j in range(0, m, batch_size):
                    X_batch = X_shuffled[j:j+batch_size]
                    Y_batch = Y_shuffled[j:j+batch_size]
                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                    if (j // batch_size + 1) % 100 == 0:
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X_batch, y: Y_batch}
                        )
                        print("\tStep {}:".format(j // batch_size + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(sess, save_path)
