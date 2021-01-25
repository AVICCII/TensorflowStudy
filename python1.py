import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random.uniform([1]))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data+biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.compat.v1.global_variables_initializer()
### create tensorflow structure end ###

sess = tf.compat.v1.Session()
sess.run(init)