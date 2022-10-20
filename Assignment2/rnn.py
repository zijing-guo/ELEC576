import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# Load MNIST dataset
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

tf.reset_default_graph()
sess = tf.InteractiveSession()



learningRate = 1e-4
trainingIters = 1e4
batchSize = 50
displayStep = 100

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 10 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, nInput])
	x = tf.split(0, nSteps, x) #configuring so you can get it as needed for the 28 pixels

	lstmCell = #find which lstm to use in the documentation

	outputs, states = #for the rnn where to get the output and hidden state

	return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = [1]))
optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	step = 1

	while step * batchSize < trainingIters:
		batchX, batchY = mnist.train.next_batch(batchSize) #mnist has a way to get the next batch
		batchX = batchX.reshape((batchSize, nSteps, nInput))

		sess.run(optimizer, feed_dict={})

		if step % displayStep == 0:
			acc = accuracy.eval(feed_dict={
                x: batchX, y: batchY})
			loss = cost.eval(feed_dict={
                x: batchX, y: batchY})
			print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
                  "{:.6f}".format() + ", Training Accuracy= " + \
                  "{:.5f}".format())
		step +=1
	print('Optimization finished')

	testData = mnist.test.images.reshape((-1, nSteps, nInput))
	testLabel = mnist.test.labels
	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={}))