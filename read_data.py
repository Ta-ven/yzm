import os
import tensorflow as tf
import pandas as pd
import settings
import numpy as np


class Reader:
    def __init__(self):
        self.img_path = settings.img_path
        self.ture_captcha_path = settings.ture_captcha_path
        self.image_suffix = settings.image_suffix
        self.ture_captcha_path = self.ture_captcha_path
        self.file_names_ls = os.listdir(self.img_path)
        self.max_captcha = settings.max_captcha
        self.char_set = settings.char_set
        self.char_set_len = len(settings.char_set)

    def confirm_image_suffix(self):
        # 在训练前校验所有文件格式
        print("开始校验所有图片后缀")
        for img_name in self.file_names_ls:
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
        file_queue = tf.train.string_input_producer(self.file_names_ls)
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

    def parse_csv(self):
        """
        读取验证码文件， 并转成oneHot
        :return: np
        """
        if '.csv' in self.ture_captcha_path:
            if settings.model != 4:
                csv_data = pd.read_csv(self.ture_captcha_path, names=["file_num", "value"], index_col="file_num")
                vector = np.zeros(self.max_captcha * self.char_set_len)
                for label in csv_data["value"]:
                    for i, ch in enumerate(label):
                        idx = i * self.char_set_len + self.char_set.index(ch)
                        vector[idx] = 1
                csv_data["labels"] = vector

            elif settings.model == 4:
                csv_data = pd.read_csv(self.ture_captcha_path, names=["file_num", "value"], index_col="file_num")
                vector = np.zeros(self.max_captcha * self.char_set_len)
                ls = []
                for label in csv_data["value"]:
                    if "=" in label:
                        label = label.strip("=")
                    for operator in ["+", "-", "×", "÷"]:
                        if operator in label:
                            a, b = label.split(operator)
                            ls = [a, operator, b]
                            break
                    for i, ch in enumerate(ls):
                        idx = i * self.char_set_len + self.char_set.index(ch)
                        vector[idx] = 1
                csv_data["labels"] = vector

            else:
                raise Exception("模式暂时不支持！")
            return csv_data

        elif ".txt" in self.ture_captcha_path:
            pass
        else:
            raise Exception("文件格式错误！")
        return None

    def filename2label(self, filename, csv_data):
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
