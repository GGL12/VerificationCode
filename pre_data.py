import os
import numpy as np
import random
from captcha.image import ImageCaptcha

def targetStr(digits,operators):
    '''
    创建随机验证码字符生成器
    '''
    seq = ''
    k = random.randint(1, 2)
    
    if k == 1:
        seq += '('
    seq += random.choice(digits)
    seq += random.choice(operators)
    if k == 2:
        seq += '('
    seq += random.choice(digits)
    if k == 1:
        seq += ')'
    seq += random.choice(operators)
    seq += random.choice(digits)
    if k == 2:
        seq += ')'
    
    return seq

def xTrain(target_str,width=180, height=60):
    '''
    生成含背景图的X_train数据
    '''
    generator = ImageCaptcha(width=width, height=height,
                             font_sizes=range(35, 56), 
                             fonts=['fonts/%s'%x for x in os.listdir('fonts') if '.tt' in x])
    generator.generate_image(target_str)
    data = np.array(generator.generate_image(target_str)).transpose(1, 0, 2)
    return data

def gen(width, height,digits,operators,rnn_length,n_len,batch_size=128):
    '''
    生成含train,label的迭代数据
    '''
    characters = digits + operators + '()'
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    while True:
        for i in range(batch_size):
            random_str = targetStr(digits,operators)
            X[i] = xTrain(random_str,width=180, height=60)
            y[i] = [characters.find(x) for x in random_str]
        yield [X, y, np.ones(batch_size)*rnn_length, np.ones(batch_size)*n_len], np.ones(batch_size)
