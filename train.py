import tensorflow as tf
import settings
from CNN import CNN
from read_data import Reader
FLAG = settings.FLAG


class Train(CNN):
    def __init__(self):
        super(Train, self).__init__()
        self.image_height = settings.image_height
        self.image_width = settings.image_width
        self.max_captcha = settings.max_captcha
        self.char_set = settings.char_set
        self.w_alpha = settings.w_alpha
        self.b_alpha = settings.b_alpha
        self.reader = Reader()
        self.model_save_dir = settings.model_save_dir

    def runner(self):
        # 构造模型
        y_predict = self.model()
        # 计算概率 损失
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=self.Y))

        # 梯度下降
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        # 计算准确率
        predict = tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len])  # 预测结果
        max_idx_p = tf.argmax(predict, 2)  # 预测结果
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.max_captcha, self.char_set_len]), 2)  # 标签
        # 计算准确率
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy_char_count = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        if FLAG == "1":
            print("开始训练！")
            with tf.Session() as sess:
                sess.run(init)
                for i in range(settings.cycle_stop):
                    images, labels_value = self.reader.get_batch(i, size=128)
                    # 梯度下降训练
                    _, cost_ = sess.run([optimizer, cost],
                                        feed_dict={self.X: images, self.Y: labels_value, self.keep_prob: 0.75})
                    acc_char = sess.run(accuracy_char_count,
                                        feed_dict={self.X: images, self.Y: labels_value, self.keep_prob: 1.})
                    # if i/20 == 0:
                    print("第%d次训练，正确率为%f，损失值为%f。" % (i, acc_char, cost_))

                    if i%200 == 0 and i != 0:
                        saver.save(sess, self.model_save_dir)
                        print("保存模型成功！")

                saver.save(sess, self.model_save_dir)
                print("保存模型成功！")
        elif FLAG == "0":
            print("开始测试!")
            with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess, self.model_save_dir)
                predict = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)
                images = self.reader.test_get_batch(r"C:\Users\Administrator\Desktop\aa", "206.jpg")
                y_pre = sess.run(predict, feed_dict={self.X: images, self.keep_prob: 1.})
                y_p = ''
                for i in y_pre[0].tolist():
                    y_p += self.char_set[int(i)]
                print("预测值为%s" % y_p)


if __name__ == '__main__':
    T = Train()
    T.runner()
