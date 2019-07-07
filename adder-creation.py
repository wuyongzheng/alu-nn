import tensorflow as tf
import numpy as np
import random

BITS = 32
HID = BITS * 3
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

def assign_op(tw1, tb1, tw2, tb2):
    w1 = np.zeros((BITS * 2, HID))
    b1 = np.zeros((HID))
    w2 = np.zeros((HID, BITS + 1))
    b2 = np.zeros((BITS + 1))

    for i1 in range(BITS):
        for i0 in range(i1+1):
            w1[i0, i1*3+0] = w1[BITS+i0, i1*3+0] = 1 << i0
            w1[i0, i1*3+1] = w1[BITS+i0, i1*3+1] = 1 << i0
            w1[i0, i1*3+2] = w1[BITS+i0, i1*3+2] = 1 << i0
            b1[i1*3+0] = 0.5 - (1 << i1)
            b1[i1*3+1] = 0.5 - (1 << i1) * 2
            b1[i1*3+2] = 0.5 - (1 << i1) * 3

    for i in range(BITS):
        w2[i*3+0, i] = 1
        w2[i*3+1, i] = -1
        w2[i*3+2, i] = 1
        b2[i] = -1.5
    w2[(BITS-1)*3+1, BITS] = 1
    b2[BITS] = -0.5

    return (tf.assign(tw1, w1), tf.assign(tb1, b1), tf.assign(tw2, w2), tf.assign(tb2, b2))

def main():
    # build graph
    x = tf.placeholder('float32', shape=(None, BITS * 2), name='input')
    y = tf.placeholder('float32', shape=(None, BITS + 1), name='label')

    w1 = tf.Variable(tf.random_normal((BITS * 2, HID)), name='w1')
    b1 = tf.Variable(tf.random_normal((HID,)), name='b1')
    hp = tf.matmul(x, w1) + b1
    h = tf.sigmoid(hp)
    #h = tf.sigmoid(tf.matmul(x, w1) + b1)

    w2 = tf.Variable(tf.random_normal((HID, BITS + 1)), name='w2')
    b2 = tf.Variable(tf.random_normal((BITS + 1,)), name='b2')
    pred = tf.sigmoid(tf.matmul(h, w2) + b2)

    error = tf.losses.mean_squared_error(y, pred)
    errori = tf.reduce_sum(tf.squared_difference(tf.round(pred), y))
    opt = tf.train.AdamOptimizer(0.01).minimize(error)

    init = tf.initialize_all_variables()
    assign = assign_op(w1, b1, w2, b2)
    saver = tf.train.Saver()
    logger = open('adder.log', 'w')

    with tf.Session() as sess:
        sess.run(init)
        sess.run(assign)
        #print(sess.run(b2))
        xv = np.zeros((1, BITS * 2))
        xv[:,0:BITS] = 1
        print(sess.run((hp, h, pred), feed_dict={x: xv}))

if __name__ == '__main__':
    main()
