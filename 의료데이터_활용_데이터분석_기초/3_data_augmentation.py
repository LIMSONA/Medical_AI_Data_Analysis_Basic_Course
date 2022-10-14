#이미지 증강 기법(Data Augmentation)을 이용하여 이미지 수 늘리기

# CNN 생성에 필요한 Keras 라이브러리, 패키지 
from keras.models import Sequential 
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D 
from keras.layers import Flatten 
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf

def create_classifier():
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


def generate_dataset(rescale_ratio, horizontal_flip):
    train_datagen = ImageDataGenerator(rescale = rescale_ratio,
                                       horizontal_flip = horizontal_flip)

    test_datagen = ImageDataGenerator(rescale = rescale_ratio)

    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size = (64, 64),
                                                     batch_size = 2,
                                                     class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size = (64, 64),
                                                batch_size = 2,
                                                class_mode = 'binary')
                                                
    return train_datagen, test_datagen, training_set, test_set


def main():
    classifier = create_classifier()

    '''지시사항 1. `rescale_ratio`의 값을 1/255로 변경하세요. '''
    rescale_ratio = 1/255 # 픽셀 값을 0-1 범위로 조정하여 표준화 => 모델 학습 속도 저하방지
    
    '''지시사항 2. `horizontal_flip`을 사용하도록 변경하세요. '''
    horizontal_flip = True # 50%의 확률로 이미지를 뒤집습니다 -> 이미지 데이터를 두 배늘리기
    
    train_datagen, test_datagen, training_set, test_set = generate_dataset(rescale_ratio, horizontal_flip)

    classifier.fit_generator(training_set,
                             steps_per_epoch = 10,
                             epochs = 10,
                             validation_data = test_set,
                             validation_steps = 10)

    output = classifier.predict_generator(test_set, steps=5)
    print(test_set.class_indices)
    
    return train_datagen, test_datagen, training_set, test_set, classifier, output
    
if __name__ == '__main__':
    main()
