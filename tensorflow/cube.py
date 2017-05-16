import tensorflow as tf

def cube():
    x = tf.placeholder(tf.float32)
    result = x**3
    return x, result

with tf.Session() as sess:
    x, result = cube()
    print(sess.run([result], {x: 3.0}))
