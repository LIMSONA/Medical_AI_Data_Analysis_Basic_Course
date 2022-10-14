# 실습에서 사용하는 라이브러리/함수
# PIL(Python Imaging Library): 이미지 분석 및 처리를 쉽게 할 수 있도록 돕는 라이브러리
# Numpy: 행렬과 같은 대규모 다차원 배열을 쉽게 처리할 수 있도록 돕는 라이브러리

import  pandas as pd
import numpy as np

import PIL
import matplotlib.image as img
import matplotlib.pyplot as plt

from elice_utils import EliceUtils
elice_utils = EliceUtils()

# 이미지 목록을 불러오는 함수입니다.

def load_data(path):
    return pd.read_csv(path)


def load_images(path, names):
    
    images=[]
    
    for name in names:
        images.append(PIL.Image.open(path+name))
    
    return images


'''
이미지의 사이즈를 main 함수에 있는 'IMG_SIZE'로 조정하고, 
이를 Numpy 배열로 변환하는 함수를 완성합니다.
'''


def images2numpy(images, size):
    
    output=[]
    
    for img in images:
        img = img.resize(size)
        np_img = np.array(img)
        output.append(np_img)     
    
    return output



# 이미지에 대한 정보를 나타내주는 함수입니다.

def sampleVisualize(np_images):

    fileName = "./data/images/1000092795.jpg"

    ndarray = img.imread(fileName)
    
    plt.imshow(ndarray)
    plt.show()    
    plt.savefig("plot.png")
    
    elice_utils.send_image("plot.png")
    print('\n1-1. Numpy array로 변환된 원본 이미지의 크기:', np.array(ndarray).shape)
    print('\n1-2. Numpy array로 변환된 resize 후 이미지 크기:', np.array(np_images[0]).shape)   
    
    plt.imshow(np_images[0])
    plt.show()
    plt.savefig("plot_re.png")
    
    elice_utils.send_image("plot_re.png")
    print('\n2-1. Numpy array로 변환된 원본 이미지:', ndarray)
    print('\n2-2. Numpy array로 변환된 resize 후 이미지 행렬:', np_images[0])



def main():

    ''' 지시사항1. 이미지 파일 경로를 변경해 이미지를 불러오세요 '''

    CSV_PATH = "./data/data.csv"
    IMG_PATH = "./data/images/"

    '''지시사항2. 이미지 크기를 조정하여 절반으로 줄이세요 '''
    IMG_SIZE = (150,150)
    MAX_LEN = 30
    BATCH_SIZE = 2
    
    name_caption = load_data(CSV_PATH)
    names = name_caption['file_name']
    
    images = load_images(IMG_PATH,names)
    np_images = images2numpy(images, IMG_SIZE)
    
    sampleVisualize(np_images)
    
    return images, np_images
    
if __name__=='__main__':
    main()