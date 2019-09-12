import tensorflow as tf
import glob
import pandas as pd
import numpy as np


# yy = sys.argv[1]
# print(sys.argv)
yy = '1'
# 定义filter和偏置
def create_weights(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape,))


def read_img():
    """
    读取图片数据
    :return:
    """

    # 1.构造文件名队列
    #     获取文件名列表
    file_names = glob.glob(r"E:\tf\yanzhengma\yzm\image\*0.jpg")
    file_queue = tf.train.string_input_producer(file_names)
    # 2.读取与解码
    reader = tf.WholeFileReader()
    filename, image = reader.read(file_queue)
    # 解码
    decoded = tf.image.decode_jpeg(image)
    # 更新形状，确定形状方便批处理
    decoded.set_shape([32, 90, 3])
    image_cast = tf.cast(decoded, tf.float32)
    # 3.批处理
    filename_batch, image_batch = tf.train.batch([filename, image_cast], batch_size=100, num_threads=1, capacity=100)
    return filename_batch, image_batch


def parse_csv():
    """
    读取目标值，解析csv
    :return:
    """
    operator_num = {"+": 100, "-": 101, "×": 102, "÷": 103}
    csv_data = pd.read_csv(r"E:\tf\yanzhengma\yzm\text.csv", names=["file_num", "value"], index_col="file_num")
    # 构造机器学习数组
    labels = []
    for label in csv_data["value"]:
        label = label.strip("=")
        for operator in ["+", "-", "×", "÷"]:
            if operator in label:
                a, b = label.split(operator)
                labels.append([int(a), operator_num[operator], int(b)])
                break
    csv_data["labels"] = labels
    return csv_data


def filename2label(filename, csv_data):
    """
    特征值和目标值一一对应
    :param filename:
    :param csv_data:
    :return:
    """
    labels = []
    for file_name in filename:
        file_num = "".join(list(filter(str.isdigit, str(file_name))))
        target = csv_data.loc[int(file_num), "labels"]
        labels.append(target)
    return np.array(labels)


def create_model(X, ):
    """
    构建神经网络
    :x:shape=[None, 30, 100, 3]
    :return:
    """
    with tf.variable_scope("conv1"):
        x = tf.reshape(X, shape=[-1, 32, 90, 3])
        # 卷积层1
        wc1 = tf.get_variable(name='wc1', shape=[3, 3, 3, 32], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc1 = tf.Variable(0.01*tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, 0.75)

        # 卷积层2
        wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc2 = tf.Variable(0.01*tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, 0.75)

        # 卷积层3
        wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc3 = tf.Variable(0.01*tf.random_normal([128]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, 0.75)
        print(">>> convolution 3: ", conv3.shape)
        next_shape = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]

        # 全连接层1
        wd1 = tf.get_variable(name='wd1', shape=[next_shape, 1024], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bd1 = tf.Variable(0.01*tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        dense = tf.nn.dropout(dense, 0.75)

        # 全连接层2
        wout = tf.get_variable('name', shape=[1024, 3 * 104], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        bout = tf.Variable(tf.random_normal([3*104]))

        with tf.name_scope('y_prediction'):
            y_predict = tf.add(tf.matmul(dense, wout), bout)

        return y_predict


if __name__ == '__main__':
    filename, image = read_img()
    csv_data = parse_csv()
    # 准备数据

    x = tf.placeholder(tf.float32, shape=[None, 32, 90, 3])
    y_ture = tf.placeholder(tf.float32, shape=[None, 3*104])
    # 构建模型
    y_predict = create_model(x)
    # 构造损失函数
    loss_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_ture, logits=y_predict)
    loss = tf.reduce_mean(loss_list)
    # 优化损失
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    # 计算准确率
    a = tf.argmax(tf.reshape(y_predict, shape=[-1, 3, 104]), axis=2)
    b = tf.argmax(tf.reshape(y_ture, shape=[-1, 3, 104]), axis=2)
    equal_list = tf.equal(a, b)
    accurary = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    init = tf.global_variables_initializer()
    # 开启回话

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        # 开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        filename_value, image_value = sess.run([filename, image])
        labels = filename2label(filename_value, csv_data)
        # 将标签值装换成one_hot
        labels_value = tf.reshape(tf.one_hot(labels, depth=104), shape=[-1, 3*104]).eval()
        if yy == '0':
            for i in range(250):
                _optimizer, error, accurary_value = sess.run([optimizer, loss, accurary], feed_dict={x: image_value, y_ture:labels_value,})
                print("第%d次训练， 损失值为：%f, 准确率为：%f" % (i+1, error, accurary_value))
                saver.save(sess, r"E:\tf\yanzhengma\model\models")
        elif yy == '1':
            saver.restore(sess, r"E:\tf\yanzhengma\model\models")
            for i in range(50):
                y_pre = sess.run(y_predict, feed_dict={x: image_value, y_ture:labels_value})
                y_pres = sess.run(tf.argmax(tf.reshape(y_pre, shape=[-1, 3, 104]), axis=-1))
                y_trues = sess.run(tf.argmax(tf.reshape(labels_value, shape=[-1, 3, 104]), axis=-1))
                equal_list = tf.reduce_all(tf.equal(y_pres, y_trues), axis=1)
                accurary = tf.reduce_mean(tf.cast(equal_list, tf.float32))
                print(sess.run(accurary))
                # print("real:%s, logist:%s" % ())
        # 回收线程
        coord.request_stop()
        coord.join(threads)