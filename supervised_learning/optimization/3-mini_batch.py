#!/usr/bin/env python3
"""
Mini-Batch
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    trains a loaded neural network model using mini-batch gradient descent:
    X_train is a numpy.ndarray of shape (m, 784) containing the training data
    Y_train is a one-hot numpy.ndarray of shape (m, 10) containing
        the training labels
    X_valid is a numpy.ndarray of shape (m, 784) containing the validation data
    Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing
        the validation labels
    batch_size is the number of data points in a batch
    epochs is the number of times the training should pass through
        the whole dataset
    load_path is the path from which to load the model
    save_path is the path to where the model should be saved after training
    Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        m = X_train.shape[0]

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
                # Shuffle before each epoch
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                # Iterate through batches
                count = 0
                for j in range(0, m, batch_size):
                    count += 1
                    end = j + batch_size
                    if end > m:
                        end = m
                    X_batch = X_shuffled[j:end]
                    Y_batch = Y_shuffled[j:end]

                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    if count > 0 and count % 100 == 0:
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X_batch, y: Y_batch}
                        )
                        print("\tStep {}:".format(count))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(sess, save_path)
