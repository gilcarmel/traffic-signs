# Load pickled data
import pickle

import cv2
import matplotlib
from tensorflow.contrib.learn.python.learn.datasets import mnist

matplotlib.use('TkAgg')
import tensorflow as tf
import numpy as np

# Original data:
training_file = "traffic-signs-data/train.p"
testing_file = "traffic-signs-data/test.p"
# Data transformed to feed the neural net
transformed_training_file = "traffic-signs-data/xformed_train.p"
transformed_testing_file = "traffic-signs-data/xformed_test.p"


def to_flattened_greyscale(image_list):
    return [np.reshape(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), [1024]) for i in image_list]


def get_or_create_transformed_data(xformed_file, orig_file):
    global n_classes
    try:
        with open(xformed_file, mode='rb') as f:
            data = pickle.load(f)
            n_classes = data['labels'][0].shape[0]
    except:
        with open(orig_file, mode='rb') as f:
            # how many classes are in the dataset
            data = pickle.load(f)
            n_classes = len(set(data['labels']))
            data['features'] = to_flattened_greyscale(data['features'])
            data['labels'] = mnist.dense_to_one_hot(data['labels'], n_classes)
            pickle.dump(data, open(xformed_file, "wb"))
    return data

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# sss = mnist.train.next_batch(128)

train = get_or_create_transformed_data(transformed_training_file, training_file)
test = get_or_create_transformed_data(transformed_testing_file, testing_file)


X_train, y_train = np.array(train['features']), np.array(train['labels'])
X_test, y_test = test['features'], test['labels']

### To start off let's do a basic data summary.

# number of training examples
n_train = len(X_train)

# number of testing examples
n_test = len(X_test)

# what's the shape of an image?
image_shape = X_train[0].shape


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Histogram of sign types
plt.figure()
n, bins, patches = plt.hist(y_train, bins=43, alpha=0.5, label="training")
n, bins, patches = plt.hist(y_test, bins=43, alpha=0.5, label="test")
plt.title("Sign counts")
plt.xlabel("Sign type")
plt.ylabel("Count")
plt.legend(loc='upper right')
plt.show()

# # heatmap of sign locations in image
# plt.figure()
# heatmap = np.zeros([image_shape[0], image_shape[1]], dtype=int)
# for size, coords in zip(train['sizes'][:50], train['coords'][:50]):
#     scale_matrix = np.mat([[32. / float(size[0]), 0.],
#                            [0., 32. / float(size[1])]])
#     coords = coords.reshape([2, 2])
#     normalized_coords = np.dot(coords, scale_matrix)
#     for x in range(int(normalized_coords[0, 0]), int(normalized_coords[1, 0])):
#         for y in range(int(normalized_coords[0, 1]), int(normalized_coords[1, 1])):
#             heatmap[x, y] += 1
#
# plt.imshow(heatmap, cmap='hot')
# plt.title("Sign location heatmap")
# plt.show()
#
# x = 4

n_input = 32 * 32  # input image pixels

# CNN parameters
# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8 * 8 * 64, 1024])),
    # 1024 inputs, 43 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def cnn():
    # Parameters
    learning_rate = 0.001
    training_iters = 200000
    batch_size = 128
    display_step = 10

    # Network Parameters
    dropout = 0.75  # Dropout, probability to keep units

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print(
                    "Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                    "{:.6f}".format(loss) + ", Training Accuracy= " +
                    "{:.5f}".format(acc)
                )
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for 256 mnist test images
        print(
            "Testing Accuracy:",
            sess.run(accuracy, feed_dict={x: X_test[:256],
                                          y: y_test[:256],
                                          keep_prob: 1.})
        )


current_index = n_train
def next_batch(batch_size):
    global current_index, X_train, y_train
    assert batch_size <= n_train
    if current_index + batch_size >= n_train:
        current_index = 0
        perm = np.arange(n_train)
        np.random.shuffle(perm)
        X_train = X_train[perm]
        y_train = y_train[perm]
    batch_x, batch_y = X_train[current_index:current_index + batch_size], y_train[current_index:current_index + batch_size]
    current_index += batch_size
    return batch_x, batch_y


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 32, 32, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

cnn()
