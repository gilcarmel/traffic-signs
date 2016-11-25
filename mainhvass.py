# Load pickled data
import pickle
from datetime import timedelta

import cv2
import time

from sklearn.metrics import confusion_matrix
from tensorflow.contrib.learn.python.learn.datasets import mnist

import tensorflow as tf
import numpy as np

# Original data:
training_file = "traffic-signs-data/train.p"
testing_file = "traffic-signs-data/test.p"
# Data transformed to feed the neural net
transformed_training_file = "traffic-signs-data/xformed_train_hvass.p"
transformed_testing_file = "traffic-signs-data/xformed_test_hvass.p"

import scipy.misc
import os


def save_images(images, path, cmin=0.0, cmax=255.0):
    index = 0
    if not os.path.exists(path):
        os.makedirs(path)
    for image in images:
        scipy.misc.toimage(image.reshape(image_shape), cmin=cmin, cmax=cmax).save(path + str(index) + '.jpg')
        index += 1


def to_flattened_greyscale(image_list):
    return [np.reshape(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), [1024])/255. for i in image_list]


def normalize_image(image):
    max_pixel = np.amax(image)
    min_pixel = np.amin(image)
    return (image - min_pixel)/(max_pixel - min_pixel)


def to_flattened_rgb(image_list):
    return [normalize_image((np.reshape(i, [1024, 3]))/255.) for i in image_list]

def get_or_create_transformed_data(xformed_file, orig_file):
    global n_classes, image_shape
    try:
        with open(xformed_file, mode='rb') as f:
            data = pickle.load(f)
            n_classes = len(set(data['labels']))
    except:
        with open(orig_file, mode='rb') as f:
            # how many classes are in the dataset
            data = pickle.load(f)
            n_classes = len(set(data['labels']))
            image_shape = data['features'][0].shape
            save_images(data['features'][:512], "traffic-signs-data/images/" + orig_file + "/")
            data['features'] = to_flattened_rgb(data['features'])
            save_images(data['features'][:512], "traffic-signs-data/images_norm/" + orig_file + "/", 0.0, 1.0)
            data['one_hot'] = mnist.dense_to_one_hot(data['labels'], n_classes)
            pickle.dump(data, open(xformed_file, "wb"))
    return data

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# sss = mnist.train.next_batch(128)

train = get_or_create_transformed_data(transformed_training_file, training_file)
test = get_or_create_transformed_data(transformed_testing_file, testing_file)


X_train, y_train, y_train_classes = np.array(train['features'], dtype=np.float32), np.array(train['one_hot'], dtype=np.float32), np.array(train['labels'])
X_test, y_test, y_test_classes = np.array(test['features'], dtype=np.float32), np.array(test['one_hot'], dtype=np.float32), np.array(test['labels'])



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

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import numpy as np

n_input = 32 * 32  # input image pixels

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# We know that sign images are 32 pixels in each dimension.
img_size = 32

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 3

# Number of classes, one class for each of 10 digits.
num_classes = n_classes

#
# def plot_images(images, cls_true, cls_pred=None):
#     assert len(images) == len(cls_true) == 9
#
#     # Create figure with 3x3 sub-plots.
#     fig, axes = plt.subplots(3, 3)
#     fig.subplots_adjust(hspace=0.3, wspace=0.3)
#
#     for i, ax in enumerate(axes.flat):
#         # Plot image.
#         ax.imshow(images[i].reshape(img_shape), cmap='binary')
#
#         # Show true and predicted classes.
#         if cls_pred is None:
#             xlabel = "True: {0}".format(cls_true[i])
#         else:
#             xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
#
#         # Show the classes as the label on the x-axis.
#         ax.set_xlabel(xlabel)
#
#         # Remove ticks from the plot.
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     # Ensure the plot is shown correctly with multiple plots
#     # in a single Notebook cell.
#     plt.show()

# Get the first images from the test-set.
images = X_test[0:9]

# Get the true classes for those images.
cls_true = np.argmax(y_test[0:9], axis=1)

# Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer



x = tf.placeholder(tf.float32, shape=[None, img_size_flat, num_channels], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

print("conv1: ", layer_conv1)

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

print("conv2: ", layer_conv2)

layer_flat, num_features = flatten_layer(layer_conv2)

print("flat: ", layer_flat)

print("num_features: ", num_features)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

layer_fc1 = tf.nn.dropout(layer_fc1, keep_prob)

print("fc1: ", layer_fc1)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

print("fc2: ", layer_fc2)

y_pred = tf.nn.softmax(layer_fc2)


y_pred_cls = tf.argmax(y_pred, dimension=1)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()

session.run(tf.initialize_all_variables())

train_batch_size = 64

# Counter for total number of iterations performed so far.
total_iterations = 0

current_index = n_train
def next_batch(batch_size):
    assert batch_size <= n_train
    global current_index, X_train, y_train, y_train_classes
    if current_index + batch_size >= n_train:
        current_index = 0
        perm = np.arange(n_train)
        np.random.shuffle(perm)
        X_train = X_train[perm]
        y_train = y_train[perm]
        y_train_classes = y_train_classes[perm]
    batch_x, batch_y = X_train[current_index:current_index + batch_size], y_train[current_index:current_index + batch_size]
    current_index += batch_size
    return batch_x, batch_y


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    dropout = 0.55

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch,
                           keep_prob: dropout}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
    pass
    # # This function is called from print_test_accuracy() below.
    #
    # # cls_pred is an array of the predicted class-number for
    # # all images in the test-set.
    #
    # # correct is a boolean array whether the predicted class
    # # is equal to the true class for each image in the test-set.
    #
    # # Negate the boolean array.
    # incorrect = (correct == False)
    #
    # # Get the images from the test-set that have been
    # # incorrectly classified.
    # images = X_test[incorrect]
    #
    # # Get the predicted classes for those images.
    # cls_pred = cls_pred[incorrect]
    #
    # # Get the true classes for those images.
    # cls_true = y_test_classes[incorrect]
    #
    # # Plot the first 9 images.
    # plot_images(images=images[0:9],
    #             cls_true=cls_true[0:9],
    #             cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    pass
#     # This is called from print_test_accuracy() below.
#
#     # cls_pred is an array of the predicted class-number for
#     # all images in the test-set.
#
#     # Get the true classifications for the test-set.
#     cls_true = y_test_classes
#
#     # Get the confusion matrix using sklearn.
#     cm = confusion_matrix(y_true=cls_true,
#                           y_pred=cls_pred)
#
#     # Print the confusion matrix as text.
#     print(cm)
#
#     # Plot the confusion matrix as an image.
#     plt.matshow(cm)
#
#     # Make various adjustments to the plot.
#     plt.colorbar()
#     tick_marks = np.arange(num_classes)
#     plt.xticks(tick_marks, range(num_classes))
#     plt.yticks(tick_marks, range(num_classes))
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#
#     # Ensure the plot is shown correctly with multiple plots
#     # in a single Notebook cell.
#     plt.show()

# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(X_test)
    # num_test = 512

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = X_test[i:j, :]

        # Get the associated labels.
        labels = y_test[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels,
                     keep_prob: 1.}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = np.argmax(y_test[:len(cls_pred)], axis=1)

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # # Plot some examples of mis-classifications, if desired.
    # if show_example_errors:
    #     print("Example errors:")
    #     plot_example_errors(cls_pred=cls_pred, correct=correct)
    #
    # # Plot the confusion matrix, if desired.
    # if show_confusion_matrix:
    #     print("Confusion Matrix:")
    #     plot_confusion_matrix(cls_pred=cls_pred)


print_test_accuracy()

optimize(num_iterations=1)

print_test_accuracy()

optimize(num_iterations=99) # We already performed 1 iteration above.

print_test_accuracy(show_example_errors=True)

optimize(num_iterations=900) # We performed 100 iterations above.


print_test_accuracy(show_example_errors=True)

for i in np.arange(28):
    optimize(num_iterations=1000) # We performed 1000 iterations above.
    print_test_accuracy(show_example_errors=True,
                        show_confusion_matrix=True)
