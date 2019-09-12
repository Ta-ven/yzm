import glob

import tensorflow as tf
import pandas
import settings


class Reader:
    def __init__(self):
        self.img_path = settings.img_path
        self.ture_captcha_path = settings.ture_captcha_path
        self.image_suffix = settings.image_suffix
    def confirm_image_suffix(self):
        # 在训练前校验所有文件格式
        print("开始校验所有图片后缀")
        for index, img_name in enumerate(self.train_images_list):
            print("{} image pass".format(index), end='\r')
            if not img_name.endswith(self.image_suffix):
                raise Exception("%s图片格式不一致" % img_name)
        print("所有图片格式校验通过")

    def read_img(self):
        """
        读取图片数据
        :return:
        """
        # 1.构造文件名队列
        #     获取文件名列表
        file_names = glob.glob(self.img_path)
        file_queue = tf.train.string_input_producer(file_names)
        # 2.读取与解码
        reader = tf.WholeFileReader()
        filename, image = reader.read(file_queue)
        # 解码
        decoded = tf.image.decode_jpeg(image)
        # 更新形状，确定形状方便批处理
        decoded.set_shape([""])
        image_cast = tf.cast(decoded, tf.float32)
        # 3.批处理
        filename_batch, image_batch = tf.train.batch([filename, image_cast], batch_size=100, num_threads=1,
                                                     capacity=200)
        return filename_batch, image_batch