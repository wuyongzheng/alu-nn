import tensorflow as tf
import numpy as np
import random

BITS = 32
HID = BITS * 3
BATCH = 1000
LR = 0.01

def gen_data():
    x = np.zeros((BATCH, BITS * 2))
    y = np.zeros((BATCH, BITS + 1))
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

def get_parameters():
    w1 = np.zeros((BITS * 2, HID), dtype=np.float32)
    b1 = np.zeros((HID), dtype=np.float32)
    w2 = np.zeros((HID, BITS + 1), dtype=np.float32)
    b2 = np.zeros((BITS + 1), dtype=np.float32)

    for i1 in range(BITS):
        for i0 in range(i1+1):
            w1[i0, i1*3+0] = w1[BITS+i0, i1*3+0] = 1 << i0
            w1[i0, i1*3+1] = w1[BITS+i0, i1*3+1] = 1 << i0
            w1[i0, i1*3+2] = w1[BITS+i0, i1*3+2] = 1 << i0
            b1[i1*3+0] = 0.5 - (1 << i1)
            b1[i1*3+1] = 0.5 - (1 << i1) * 2
            b1[i1*3+2] = 0.5 - (1 << i1) * 3
    # let's make it larger so the pre-sigmoid value is further away from 0,
    # and so the post-sigmoid value is more distinct.
    w1 = w1 * 8
    b1 = b1 * 8

    for i in range(BITS):
        w2[i*3+0, i] = 8
        w2[i*3+1, i] = -8
        w2[i*3+2, i] = 8
        b2[i] = -4
    w2[(BITS-1)*3+1, BITS] = 8
    b2[BITS] = -4

    return w1, b1, w2, b2

def main():
    x = tf.placeholder(tf.float32, shape=(None, BITS * 2), name='input')
    y = tf.placeholder(tf.float32, shape=(None, BITS + 1), name='label')

    npw1, npb1, npw2, npb2 = get_parameters()
    w1 = tf.Variable(npw1, name='w1')
    b1 = tf.Variable(npb1, name='b1')
    h = tf.sigmoid(tf.matmul(x, w1) + b1)

    w2 = tf.Variable(npw2, name='w2')
    b2 = tf.Variable(npb2, name='b2')
    pred = tf.sigmoid(tf.matmul(h, w2) + b2)

    error = tf.losses.mean_squared_error(y, pred)
    errori = tf.reduce_sum(tf.squared_difference(tf.round(pred), y))
    opt = tf.train.AdamOptimizer(0.01).minimize(error)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(100):
            xv, yv = gen_data()
            errv, erriv = sess.run((error, errori), feed_dict={x: xv, y: yv})
            print(errv, erriv)

if __name__ == '__main__':
    main()
