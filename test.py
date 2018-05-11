from gen_captcha import gen_captcha_text_and_image

import matplotlib.pyplot as plt
from PIL import Image

from training import convert2gray
from training import vec2text
from training import crack_captcha_cnn

import time
import training as tr
import numpy as np
import tensorflow as tf


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint("./"))

        predict = tf.argmax(tf.reshape(output, [-1, tr.MAX_CAPTCHA, tr.CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={tr.X: [captcha_image], tr.keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(tr.MAX_CAPTCHA * tr.CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * tr.CHAR_SET_LEN + n] = 1
            i += 1
        return vec2text(vector)


if __name__ == '__main__':
    start = time.clock()

    text, image = gen_captcha_text_and_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 1.1, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    image = convert2gray(image)
    image = image.flatten() / 255
    predict_text = crack_captcha(image)
    print("correct: {}  predict: {}".format(text, predict_text))

    end = time.clock()
    print('Running time: %s Seconds' % (end - start))

    plt.show()
