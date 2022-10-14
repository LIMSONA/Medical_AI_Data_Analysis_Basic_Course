# Confusion Matrix를 통한 딥러닝 성능 평가하기

# 패키지 불러오기
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score

import os
import pandas as pd
import seaborn as sn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt

from elice_utils import EliceUtils
elice_utils = EliceUtils()
import model as md


def get_result(model,test_set, classNames):
    y_pred = model.predict_generator(test_set,steps=20)
    y_pred = np.argmax(y_pred, axis=1)
    # y_test에 test_set의 라벨을 0,1로 선언합니다.
    y_test = test_set.labels


    # y_prediction과 y_test를 직접 확인해보세요.
    print(test_set.class_indices)
    print("model predictions: ", y_pred)
    print("golden labels: ", y_test)


    # accuracy_score()함수를 활용하여 acc를 구합니다.
    # acc는 % 표기를 따릅니다.
    acc = accuracy_score(y_pred,y_test) * 100
    print('%.2f' % acc,'%')


    # y_test와 y_pred를 활용하여 confusion matrix를 cm으로 선언합니다.
    cm = confusion_matrix(y_test, y_pred)
    # 위에서 선언한 confusion matrix으로 dataframe에 생성합니다.


    print(cm)

    df_cm = pd.DataFrame(cm, index = [i for i in classNames], columns = [i for i in classNames])
    
    sn.set(font_scale=1.5)
    plt.figure(figsize = (10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()
    plt.savefig('graph.png')
    elice_utils.send_image('graph.png')
    mpl.rcParams.update(mpl.rcParamsDefault)
    K.clear_session() # 세션 클리어
    return y_pred, y_test, acc, cm, df_cm

def calculate(cm):
    '''지시사항 1. cm을 확인하고 위치를 입력하여 ` tp`, ` fn`, ` fp`, ` tn`을 구하세요.'''
    tp = cm[0][0]
    fn = cm[1][0]
    fp = cm[0][1]
    tn = cm[1][1]
    
    '''지시사항 2. ` tp`, ` fn`, ` fp`, ` tn`를 이용하여 accuracy, precision, recall, f-1 score를 구하세요.'''
    accuracy = (tp+tn)/(tp+fn+fp+tn) 
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(precision*recall)/(precision+recall)

    return accuracy, precision, recall, f1


def main():
    dir_loc = 'dataset/'
    dir_training = 'training_set'
    dir_test = 'test_set'
    classNames = os.listdir('./'+dir_loc+dir_test) # 각 클래스의 이름들
    numClass = len(classNames)     
    image_size = 64
    batch_size = 64

    model = md.generate_model()
    training_set, test_set = md.generate_imgset (dir_loc + dir_training, dir_loc + dir_test)

    result, model = md.train_model(model, training_set, test_set)
    
    y_pred, y_test, acc, cm, df_cm = get_result(model, test_set, classNames)

    accuracy, precision, recall, f1 = calculate(cm)

    print("accuracy: %f, precision: %f, recall: %f, f1-score: %f"%(accuracy, precision, recall, f1))


if __name__ =='__main__':
    main()