import tensorflow as tf

tf.compat.v1.disable_eager_execution()

state = tf.Variable(0, name='counter')
print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.compat.v1.assign(state, new_value)

init = tf.compat.v1.initialize_all_variables()  # must have if define variable

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
