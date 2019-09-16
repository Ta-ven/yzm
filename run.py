import settings
from working import Working
from train import Train


value = settings.FLAG

if value == '1':
    t = Train()
    t.runner()

elif value == '0':
    w = Working()
    w.runner(r"C:\Users\Administrator\Desktop\a", "229.jpg")