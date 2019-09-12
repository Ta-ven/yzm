import os
import tensorflow as tf
import pandas as pd
import settings
import numpy as np
from CNN import CNN


class Train():
    def __init__(self):
        "image_height, image_width, max_captcha, char_set, model_save_dir, w_alpha, b_alpha"
        self.image_height = settings.image_height
        self.image_width = settings.image_width
        self.max_captcha = settings.max_captcha
        self.char_set = settings.char_set
        self.w_alpha = settings.w_alpha
        self.b_alpha = settings.b_alpha
        self.cnn = CNN()

    def run(self):
        pass