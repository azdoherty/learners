import tensorflow as tf
import numpy as np
# taken from example here: http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
# added docstrings for clarity and worked comments directly into code
# datasets to playwith


def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    """
    :param input_data: images (as TF variables)
    :param num_input_channels: channels of image (1 for grayscale)
    :param num_filters: channels of second layer
    :param filter_shape: shape of the weights of the convolution filter (shape[0] x shape[1]
    :param pool_shape:
    :param name:
    :return:
    """
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation -  kernels moves 1 in both x and y
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    return out_layer

# output of layer 1 must equal input channels of layer 2

class TensorFlowNeuralNetwork:
    def __init__(self, xshape, yshape, learning_rate=.0001):
        # optimization
        self.learning_rate = 0.0001

        # declare the training data placeholders
        size = xshape * yshape
        self.x = tf.placeholder(tf.float32, [None, size])
        # dynamically reshape the input into 4d object as required by TF
        x_shaped = tf.reshape(self.x, [-1, xshape, yshape, 1])
        # now declare the output data placeholder - 10 digits possible
        self.y = tf.placeholder(tf.float32, [None, 10])

        layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
        layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
        flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])
        wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03), name='wd1')
        bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
        dense_layer1 = tf.matmul(flattened, wd1) + bd1
        dense_layer1 = tf.nn.relu(dense_layer1)
        wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
        bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
        dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
        y_ = tf.nn.softmax(dense_layer2)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=self.y))

        # add an optimiser
        self.optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cross_entropy)

        # define an accuracy assessment operation
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # setup the initialisation operator
        self.init_op = tf.global_variables_initializer()
        # setup the initialisation operator
        self.writer = tf.summary.FileWriter('output')
        tf.summary.scalar('accuracy', self.accuracy)

    def next_batch(self, batch_size, X, y):
        indices = np.random.choice(X.shape[0], batch_size, replace=False)
        return X[:, indices], y[:, indices]

    def train(self, X, y, testX, testy, epochs, batch_size):
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            # initialise the variables
            sess.run(self.init_op)
            total_batch = int(X.shape[0] / batch_size)
            for epoch in range(epochs):
                avg_cost = 0
                for i in range(total_batch):
                    batch_x, batch_y = self.next_batch(batch_size, X, y)
                    _, c = sess.run([self.optimiser, self.cross_entropy],
                                    feed_dict={self.x: batch_x, self.y: batch_y})
                    avg_cost += c / total_batch
                test_acc = sess.run(self.accuracy,
                               feed_dict={self.x: testX, self.y: testy})
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: {:.3f}".format(test_acc))
                summary = sess.run(merged, feed_dict={self.x: testX, self.y: testy})
                self.writer.add_summary(summary, epoch)
            self.writer.add_graph(sess.graph)
            print(sess.run(self.accuracy, feed_dict={self.x: testX, self.y: testy}))
            print("\nTraining complete!")
        print(sess.run(self.accuracy, feed_dict={self.x: testX, self.y: testy}))


if __name__ == "__main__":
    epochs = 5
    batch_size = 500
    nn = TensorFlowNeuralNetwork(28, 28)
    nn.train(5, 10)

