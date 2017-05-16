import tensorflow as tf

def model(learning_rate):
    x = tf.Variable(20.0)
    sq = x ** 2
    o = tf.train.GradientDescentOptimizer(learning_rate).minimize(sq)
    return x, o

with tf.Session() as sess:
    x, o = model(0.1)
    init = tf.global_variables_initializer()
    sess.run([init])
    
    for i in range(100):
        sess.run([o])

    (x,) = sess.run([x])
    print(x)
