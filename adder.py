import tensorflow as tf
import numpy as np
import random

BITS = 32
HIDW = 160
HIDH = 1
BATCH = 1000
LR = 0.01

def gen_data():
    x = np.zeros((BATCH, BITS * 2), dtype=np.float32)
    y = np.zeros((BATCH, BITS + 1), dtype=np.float32)
    for i in range(BATCH):
        r1 = random.randint(0, (1 << BITS) - 1)
        x1 = [1 if r1 & (1 << x) else 0 for x in range(BITS)]
        r2 = random.randint(0, (1 << BITS) - 1)
        x2 = [1 if r2 & (1 << x) else 0 for x in range(BITS)]
        y1 = r1 + r2
        y1 = [1 if y1 & (1 << x) else 0 for x in range(BITS+1)]
        x[i, :] = x1 + x2
        y[i, :] = y1
    return x, y

def main():
    # build graph
    x = tf.placeholder('float32', shape=(None, BITS * 2), name='input')
    y = tf.placeholder('float32', shape=(None, BITS + 1), name='label')

    w = tf.Variable(tf.random_normal([BITS * 2, HIDW]), name='w1')
    b = tf.Variable(tf.random_normal([HIDW]), name='b1')
    h = tf.sigmoid(tf.matmul(x, w) + b)

    for l in range(2, HIDH+1):
        w = tf.Variable(tf.random_normal([HIDW, HIDW]), name='w' + str(l))
        b = tf.Variable(tf.random_normal([HIDW]), name='b' + str(l))
        h = tf.sigmoid(tf.matmul(h, w) + b)

    w = tf.Variable(tf.random_normal([HIDW, BITS + 1]), name='wo')
    b = tf.Variable(tf.random_normal([BITS + 1]), name='bo')
    pred = tf.sigmoid(tf.matmul(h, w) + b)

    error = tf.losses.mean_squared_error(y, pred)
    errori = tf.reduce_sum(tf.squared_difference(tf.round(pred), y))
    opt = tf.train.AdamOptimizer(0.01).minimize(error)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    logger = open('adder.log', 'w')

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(320000):
            xv, yv = gen_data()
            _, errv, erriv, predv = sess.run((opt, error, errori, pred), feed_dict={x: xv, y: yv})
            if epoch % 1000 == 0:
                predv = np.round(predv, 2)
                print(errv, erriv, yv[0,:], predv[0,:])
                saver.save(sess, './adder-cp')
            elif epoch % 100 == 0:
                print(epoch, errv)
            logger.write('{}\t{}\t{}\n'.format(epoch, errv, erriv/BATCH))

if __name__ == '__main__':
    main()
