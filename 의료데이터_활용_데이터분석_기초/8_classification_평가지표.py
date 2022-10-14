# 분류(Classification) 평가 지표
# sklearn.metrics 서브 패키지
# roc_curve(y_true, y_pred) -> fpr, tpr
# : 실제 테스트 데이터의 y값과 모델로 예측하여 얻은 y값 사이의 관계를 파악하기 위해 값을 넣어주면, FPR(False Positive Rate), TPR(True Positive Rate)를 리턴해줍니다.

# auc(fpr, tpr) : fpr, tpr로부터 계산된 결과를 ROC Curve에 그릴 수 있도록 영역을 지정합니다.

import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from elice_utils import EliceUtils
elice_utils = EliceUtils()

dir_loc = 'dataset/'
dir_training = 'training_set'
dir_test = 'test_set'
classNames = os.listdir('./'+dir_loc+dir_test) # 각 클래스의 이름들
numClass = len(classNames)     
image_size = 64
batch_size = 64

import model as md
import result as rs


def one_hot(l):
    temp_list = list()
    for i in l:
        if i == 0:
            temp_list.append([1,0])
        else:
            temp_list.append([0,1])
    return np.array(temp_list)

def main():
    model = md.generate_model()
    training_set, test_set = md.generate_imgset (dir_loc + dir_training, dir_loc + dir_test)
    result, model = md.train_model(model, training_set, test_set)
    
    y_pred, y_test, acc, cm, df_cm = rs.get_result(model, training_set,test_set)    
    y_pred = one_hot(label_binarize(y_pred, classes = [0,1]))
    y_test = one_hot(label_binarize(y_test, classes = [0,1]))
    generate_roc(y_pred, y_test)
    return y_pred,y_test


def generate_roc(y_pred, y_test):
    '''
    지시사항 1번
        ROC curve를 사용하여 분류 성능을 확인해봅니다.
    '''
    
    # 각 클래스의 ROC Curve 값을 계산하여 넣어 줄 변수를 선언합니다.
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(numClass):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])

        # <To Do>: auc() 함수에 fpr[i], tpr[i]를 인자로 넣어 각각의 클래스에서의 ROC & AUC 값을 획득하세요.
        roc_auc[i] = auc(fpr[i], tpr[i])

    temp_cls = list()
    for nth_class in range(numClass):
        temp_cls.append(plot_ROC_curve(fpr, tpr, roc_auc, nth_class))
        
    return numClass,fpr,tpr,roc_auc,temp_cls
    
# ROC curve를 그리기 위해 사용되는 함수입니다.
def plot_ROC_curve(fpr, tpr, roc_auc, nth_class):

    plt.figure()
    lw = 2
    
    color_name = ''
    
    '''
    지시사항 2번
        class에 따라 별도의 색을 지정합니다.
    '''
    # <To Do>: if 문을 사용하여 nth_class의 인자가 normal인 경우를 산정합니다.
    #  normal인 경우 현재 0으로 코딩되어있습니다.
    if nth_class == 0:
        #<To Do>: color_name을 'red'로 선언합니다.
        color_name = "red"
    else:
        # <To Do>: 그 외의 경우, color_name을 'orange'로 선언합니다.
        color_name = "orange"
        
    plt.plot(fpr[nth_class], tpr[nth_class], color=color_name,
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[nth_class])
    plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Class %s ROC(Receiver Operating Characteristic) Curve' %nth_class)
    plt.legend(loc="lower right")
    plt.savefig('roc curve.png')
    elice_utils.send_image('roc curve.png')
    
    return color_name
    

if __name__ == "__main__":
    y_pred,y_test = main()