import numpy as np
import tensorflow as tf
import time
import os
import random
import matplotlib.pyplot as plt
import matplotlib as mp
if(tf.__version__.split('.')[0]=='2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


# --------------------------------------------------
# setup

def var_summaries(var):
    '''
    tool for tensorboard visulization
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.1)
    # initial = tf.contrib.layers.xavier_initializer()
    W =  tf.Variable(initial)
    return W


def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    return b


def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv


def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    return h_max


def convolution_layer(x, shape, name, actFun=tf.nn.relu):
    '''
    API for convolution layer
    '''
    with tf.name_scope(name):
        with tf.name_scope('input'):
            var_summaries(x)

        with tf.name_scope('weights'):
            W = weight_variable(shape)
            var_summaries(W)

        with tf.name_scope('bias'):
            b = bias_variable([shape[-1]])

        with tf.name_scope('activation'):
            z = conv2d(x, W) + b
            h = actFun(z)
            var_summaries(h)

        with tf.name_scope('pooling'):
            a = max_pool_2x2(h)
            var_summaries(a)

    return a, W


def fully_connected_layer(x, shape, name, actFun=tf.nn.relu):
    with tf.name_scope(name):
        with tf.name_scope('input'):
            x_flat = tf.reshape(x, [-1, shape[0]])
            var_summaries(x_flat)

        with tf.name_scope('weights'):
            W = weight_variable(shape)
            var_summaries(W)

        with tf.name_scope('bias'):
            b = bias_variable([shape[-1]])
            var_summaries(b)

        with tf.name_scope('activation'):
            z = tf.matmul(x_flat, W) + b
            h = actFun(z)
            var_summaries(h)

    return h


# --------------------------------------------------
# parameter setting

ntrain = 1000  # per class
ntest = 100  # per class
nclass = 10  # number of classes
imsize = 28
nchannels = 1
batchsize = 100
nsamples = ntrain * nclass

Train = np.zeros((ntrain * nclass, imsize, imsize, nchannels))
Test = np.zeros((ntest * nclass, imsize, imsize, nchannels))
LTrain = np.zeros((ntrain * nclass, nclass))
LTest = np.zeros((ntest * nclass, nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = './CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        # im = misc.imread(path);  # 28 by 28
        # im = im.astype(float) / 255
        im = plt.imread(path)
        itrain += 1
        Train[itrain, :, :, 0] = im
        LTrain[itrain, iclass] = 1  # 1-hot lable
    for isample in range(0, ntest):
        path = './CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
        # im = misc.imread(path);  # 28 by 28
        # im = im.astype(float) / 255
        im = plt.imread(path)
        itest += 1
        Test[itest, :, :, 0] = im
        LTest[itest, iclass] = 1  # 1-hot lable

# --------------------------------------------------
# model
# create your model

sess = tf.InteractiveSession()

# Specify training parameters
result_dir = './results/'  # directory where the results from the training are saved
max_step = 10000  # the maximum iterations. After max_step iterations, the training will stop no matter what

# placeholders for input data and input labeles
tf_data = tf.placeholder(tf.float32, [None, imsize, imsize, nchannels],
                         name='data_input')  # tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder(tf.float32, [None, nclass], name='label_input')  # tf variable for labels

# 1st convolution layer
h_pool1, w_pool1 = convolution_layer(tf_data, [5, 5, 1, 32], 'conv1')

# 2nd convolution layer
h_pool2, w_pool2 = convolution_layer(h_pool1, [5, 5, 32, 64], 'conv2')

# 1st fully connected layer
h_fc1 = fully_connected_layer(h_pool2, [7 * 7 * 64, 1024], 'fully_connected_1', tf.nn.relu)

# dropout
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 2nd fully connected layer
y_conv = fully_connected_layer(h_fc1_drop, [1024, nclass], 'fully_connected_2', tf.nn.softmax)

# --------------------------------------------------
# visulizing the trained network
w_pool1_splits = tf.split(w_pool1, 32, 3)
count = len(w_pool1_splits)
for i in range(count):
    tf.summary.image('filter' + str(i), tf.transpose(w_pool1_splits[i], [3, 0, 1, 2]))

# --------------------------------------------------
# loss

# set up the loss, optimization, evaluation, and accuracy
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=y_conv))
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(1e-03).minimize(cross_entropy)
with tf.name_scope('correct'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# --------------------------------------------------
# summary
summary_op = tf.summary.merge_all()
saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

# --------------------------------------------------
# optimization

start_time = time.time()  # start timing

sess.run(tf.global_variables_initializer())
batch_xs = np.zeros(
    (batchsize, imsize, imsize, nchannels))  # setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros((batchsize, nclass))  # setup as [batchsize, the how many classes]
for i in range(max_step):  # try a small iteration size once it works then continue
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j, :, :, :] = Train[perm[j], :, :, :]
        batch_ys[j, :] = LTrain[perm[j], :]

    if i % 100 == 0:
        # calculate train accuracy and print it
        train_accuracy = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        train_loss = cross_entropy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy: %g, training loss: %g" % (i, train_accuracy, train_loss))

        summary_str = sess.run(summary_op, feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

    if i % 100 == 0:
        checkpoint = os.path.join(result_dir, 'checkpoint')
        saver.save(sess, checkpoint, global_step=i)

    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})  # dropout only during training

# --------------------------------------------------
# test
stop_time = time.time()  # start timing
print('\ntraining time consumption: %f' % (stop_time - start_time))

print("test accuracy %g" % accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))
sess.close()
