FLAG = "0"  # 1是训练0是验证

# 图片基本信息
img_path = r'C:\Users\Administrator\Desktop\img\img'
ture_captcha_path = r'C:\Users\Administrator\Desktop\img\text.csv'
image_height = 30
image_width = 150
image_suffix = 'png'
channel_num = 3

# 验证码基本信息
# 验证码样式(数字 1、字母 2、数字加字母 3、四则运算 4)
model = 3
max_captcha = 5
# 字符串列表
char_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

w_alpha = 0.01
b_alpha = 0.1

# 保存路径
model_save_dir = r'C:\Users\Administrator\Desktop\img\model\model'

# 训练次数
cycle_stop = 3000