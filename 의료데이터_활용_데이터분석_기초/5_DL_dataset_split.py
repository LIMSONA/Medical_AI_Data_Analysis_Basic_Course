# 딥러닝 학습을 위한 데이터셋 분리
# iris dataset

import random 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def merge_data(data_list, label_list):

    new_dataset = {}
    # data_list = []
    # label_list = []
    # for i, d in enumerate(data):
    #     data_list.append(d)
    #     label_list.append(label[i])
    new_dataset['data'] = data_list
    new_dataset['target'] = label_list

    return new_dataset

def divide_data(data, label, split_ratio):
    train_data = data[:int(split_ratio[0]*len(data))] # [:20] [10:20]
    val_data = data[int(split_ratio[0]*len(data)):int((split_ratio[0] + split_ratio[1])*len(data))]
    test_data = data[int((split_ratio[0] + split_ratio[1])*len(data)):]

    train_label = label[:int(split_ratio[0]*len(data))]
    val_label = label[int(split_ratio[0]*len(data)):int((split_ratio[0] + split_ratio[1])*len(data))]
    test_label = label[int((split_ratio[0] + split_ratio[1])*len(data)):]

    return train_data, val_data, test_data, train_label, val_label, test_label

def split_data():
    entire_dataset = load_iris()
    data = entire_dataset['data'] # 데이터 전체(레이블제외)
    label = entire_dataset['target'] # 레이블 데이터
    
    print('원 데이터셋의 shape: ',data.shape)
    print("원 데이터: '\n' ", entire_dataset['data'][20:30])
    print('데이터셋 레이블: ', entire_dataset['target'][:10])
    

    '''
    지시사항 1번
        train_test_split() 함수를 활용하여
        train: test = 7 : 1.5 : 1.5 비율로 데이터를 나눕니다.
    '''
    split_ratio = [7/10, 1.5/10 , 1.5/10] # 전체 중의 비율로 나타내야 하니까 10으로 나눔

    split_data_1, split_data_2, split_data_3, split_label_1, split_label_2, split_label_3 = divide_data(data, label, split_ratio)

    '''
    지시사항 2번 
        merge_data() 함수를 활용하여 학습, 검증 및 테스트 데이터셋트를 구성합니다.
    '''
    train_dataset = merge_data(split_data_1, split_label_1)
    val_dataset = merge_data(split_data_2, split_label_2)
    test_dataset = merge_data(split_data_3, split_label_3)

    
    print('학습셋의 데이터 shape: ', train_dataset['data'].shape)
    print('학습셋의 레이블 shape: ', train_dataset['target'].shape)
    print('검증셋의 데이터 shape: ', val_dataset['data'].shape)
    print('검증셋의 레이블 shape: ', val_dataset['target'].shape)
    print('테스트셋의 데이터 shape: ', test_dataset['data'].shape)
    print('테스트의 레이블 shape: ', test_dataset['target'].shape)


    return split_ratio, train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    split_data()