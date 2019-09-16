import tensorflow as tf
from CNN import CNN
from read_data import Reader
import settings


class Working(CNN):
    def __init__(self):
        super(Working, self).__init__()
        self.image_height = settings.image_height
        self.image_width = settings.image_width
        self.max_captcha = settings.max_captcha
        self.char_set = settings.char_set
        self.reader = Reader()
        self.model_save_dir = settings.model_save_dir

    def runner(self, distinguish_img_path, img_name):
        y_predict = self.model()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, self.model_save_dir)
            predict = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)

            images = self.reader.test_get_batch(distinguish_img_path, img_name)
            y_pre = sess.run(predict, feed_dict={self.X: images, self.keep_prob: 1.})

            y_p = ''
            for i in y_pre[0].tolist():
                y_p += self.char_set[int(i)]
            print("预测值为%s" % y_p)
            return y_p