import os
import time
import pandas as pd
import settings
import numpy as np
from PIL import Image
import random


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
        if settings.FLAG == "1":
            self.csv_data = self.parse_csv()
            self.confirm_image_suffix()
            self.train_images_list = os.listdir(self.img_path)
            # 打乱图片
            random.seed(time.time())
            random.shuffle(self.train_images_list)

    def confirm_image_suffix(self):
        # 在训练前校验所有文件格式
        print("开始校验所有图片后缀")
        for img_name in self.file_names_ls:
            if not img_name.endswith(self.image_suffix):
                raise Exception("%s图片格式不一致" % img_name)
        print("所有图片格式校验通过")

    def read_img(self, img_path, img_name):
        """
        读取图片数据
        :return:
        """
        img_file = os.path.join(img_path, img_name)
        captcha_image = Image.open(img_file)
        captcha_array = np.array(captcha_image)  # 向量化
        num = int(img_name.split(".")[0])
        label = self.csv_data.loc[num, "labels"]
        return label, captcha_array

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
                ls = []
                for label in csv_data["value"]:
                    li = []
                    if "=" in label:
                        label = label.strip("=")
                    for operator in ["+", "-", "×", "÷"]:
                        if operator in label:
                            a, b = label.split(operator)
                            for j, ch in enumerate([a, operator, b]):
                                idx = j * self.char_set_len + self.char_set.index(ch)
                                li.append(idx)
                    ls.append(li)
                csv_data["labels"] = ls

            else:
                raise Exception("模式暂时不支持！")
            return csv_data

        elif ".txt" in self.ture_captcha_path:
            pass
        else:
            raise Exception("文件格式错误！")
        return None

    def convert2gray(self, img):
        """
        图片转为灰度图，如果是3通道图则计算，单通道图则直接返回
        :param img:
        :return:
        """
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img

    def one_hot(self, label):
        vector = np.zeros(self.max_captcha * self.char_set_len)
        for i in label:
            vector[int(i)] = 1
        return vector

    def get_batch(self, n, size=128):
        batch_x = np.zeros([size, settings.image_height * settings.image_width])  # 初始化
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len])  # 初始化

        max_batch = int(len(self.train_images_list) / size)
        if max_batch - 1 < 0:
            raise Exception("训练集图片数量需要大于每批次训练的图片数量")
        if n > max_batch - 1:
            n = n % max_batch
        s = n * size
        e = (n + 1) * size
        this_batch = self.train_images_list[s:e]

        for i, img_name in enumerate(this_batch):
            label, image_array = self.read_img(self.img_path, img_name)
            image_array = self.convert2gray(image_array)  # 灰度化图片
            batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
            batch_y[i, :] = self.one_hot(label)  # 生成 oneHot
        return batch_x, batch_y

    def test_get_batch(self, img_path, img_name):
        batch_x = np.zeros([1, settings.image_height * settings.image_width])  # 初始化
        # batch_y = np.zeros([1, self.max_captcha * self.char_set_len])  # 初始化
        image_array = self.test_read_img(img_path, img_name)
        image_array = self.convert2gray(image_array)  # 灰度化图片
        batch_x[0, :] = image_array.flatten() / 255  # flatten 转为一维
        return batch_x

    def test_read_img(self, img_path, img_name):
        """
        读取图片数据
        :return:
        """
        img_file = os.path.join(img_path, img_name)
        captcha_image = Image.open(img_file)
        captcha_array = np.array(captcha_image)  # 向量化
        return captcha_array
