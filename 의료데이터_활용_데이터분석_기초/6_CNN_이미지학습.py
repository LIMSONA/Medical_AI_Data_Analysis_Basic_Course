# 신경망 모델로 이미지 학습하기

# 패키지 불러오기
import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Model
from keras.models import Sequential 
from tensorflow.keras.layers import Conv2D 
from keras.layers import MaxPooling2D 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import seaborn as sn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import PIL
import matplotlib.image as img
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from elice_utils import EliceUtils
elice_utils = EliceUtils()
import vgg

def generate_model(image_size, numClass):
    '''
    지시사항 2-1. VGG 모델의 입력 이미지 사이즈를 224x224로 변경해주세요
    '''
    model = vgg.VGG19(input_shape=(224, 224, 3))

    model.add(Dense(numClass, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-3),
                  metrics=['acc'])

    # 새로운 모델 요약
    model.summary()
    return model

def generate_imgset(tr_loc,te_loc):
    #Training_set을 생성
    train_datagen = ImageDataGenerator(
                      rescale=1/255,
                      rotation_range=3,
                      width_shift_range=0.01,
                      height_shift_range=0.10,
                      zoom_range=0.05,
                      fill_mode='nearest')

    '''
    지시사항 2-2. 학습 및 테스트 이미지의 크기를 224x224로 변경해주세요.
    '''

    training_set = train_datagen.flow_from_directory(tr_loc,
                                                     target_size = (224, 224),
                                                     batch_size = 2,
                                                     class_mode = 'categorical')

    #test_set을 생성
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory(te_loc,
                                                target_size = (224, 224),
                                                batch_size = 2,
                                                class_mode = 'categorical')
    return training_set, test_set

def train_model(model, training_set, val_set):
    '''
    지시사항 1.
            model을 training data에 맞게 fit()합니다.
    '''

    result = model.fit(training_set, 
                             steps_per_epoch = 20,
                             epochs = 4,
                             validation_data = val_set)
    return result, model

def main():
    dir_loc = 'dataset/'
    dir_training = 'training_set'
    dir_test = 'test_set'
    classNames = os.listdir('./'+dir_loc+dir_test) # 각 클래스의 이름들
    numClass = len(classNames)     

    '''
    지시사항 2-1. VGG 모델의 입력 이미지 사이즈를 224x224로 변경해주세요.
    '''
    image_size = 224
    batch_size = 64

    model = generate_model(image_size, numClass)
    training_set, test_set = generate_imgset (dir_loc + dir_training, dir_loc + dir_test)


    result, model = train_model(model, training_set, test_set)

if __name__ =='__main__':
    main()

