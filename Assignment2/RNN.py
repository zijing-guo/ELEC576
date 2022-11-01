import tensorflow as tf
import os
import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # call mnist function

if (tf.__version__.split('.')[0] == '2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

learningRate = 1e-03
trainingIters = 6000
batchSize = 128
displayStep = 200

result_dir = './results-RNN/'  # directory where the results from the training are saved

nInput = 28  # we want the input to take the 28 pixels
nSteps = 28  # every 28
nHidden = 10 # number of neurons for the RNN
nClasses = 10  # this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
    'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}


def RNN(x, weights, biases, type='LSTM'):
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # x = tf.reshape(x, [-1, nSteps, nInput])
    x = tf.unstack(x, nSteps, 1)
    if type == 'GRU':
        cell = rnn_cell.GRUCell(nHidden)  # find which lstm to use in the documentation
    if type == 'RNN':
        cell = rnn_cell.BasicRNNCell(nHidden)
    if type == 'LSTM':
        cell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0)

    # outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)  # for the rnn where to get the output and hidden state
    outputs, states = tf.nn.static_rnn(cell, x, dtype=tf.float32)  # for the rnn where to get the output and hidden state
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(x, weights, biases)
pred = tf.nn.softmax(logits)

# optimization
# create the cost, optimization, evaluation, and accuracy
# for the cost softmax_cross_entropy_with_logits seems really good
with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    tf.summary.scalar('loss', cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

with tf.name_scope('correct'):
    correctPred = tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32)

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

    for i in range(trainingIters):
        batchX, batchY = mnist.train.next_batch(batchSize)  # mnist has a way to get the next batch
        batchX = batchX.reshape((batchSize, nSteps, nInput))
        sess.run(optimizer, feed_dict={x: batchX, y: batchY})
        if i % displayStep == 0:
            acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
            loss = sess.run(cost, feed_dict={x: batchX, y: batchY})
            print("Step " + str(i) + ", Minibatch Loss= " + "{:.6f}".format(
                loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            testData = mnist.test.images.reshape((-1, nSteps, nInput))
            testLabel = mnist.test.labels
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
            # summary writing
            summary_str = sess.run(summary_op, feed_dict={x: batchX, y: batchY})
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

        if i % (5 * displayStep) == 0:
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)


    print('Optimization finished')
    testData = mnist.test.images.reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
