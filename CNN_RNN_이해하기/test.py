# 1번 > 이미지 편집
# `crop` 메소드는 잘라낼 영역의 좌측 상단 좌표$(x_1, y_1)$과 우측 하단 좌표$(x_2, y_2)$값 4개를 튜플로 요구합니다. 즉, `crop` 메소드는 아래와 같은 형태로 사용해야 합니다.
# img.crop((x1, y1, x2, y2))

#  2번 > CNN 행렬곱
# CNN 행렬곱 -> 그자리 그대로 곱하기
# https://untitledtblog.tistory.com/150 참고

# 3번 > RNN 설명

# 4번문제 > CNN 모델로 CIFAR-10 분류하기

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from elice_utils import EliceUtils

elice_utils = EliceUtils()

import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt

def load_cifar10():
    # CIFAR-10 데이터셋을 불러옵니다.
    X_train = np.load("cifar10_train_X.npy")
    y_train = np.load("cifar10_train_y.npy")
    X_test = np.load("cifar10_test_X.npy")
    y_test = np.load("cifar10_test_y.npy")

    # TODO: [지시사항 1번] 이미지의 각 픽셀값을 0에서 1 사이로 정규화하세요.
    X_train = X_train/255.0
    X_test = X_test/255.0
    
    # 정수 형태로 이루어진 라벨 데이터를 one-hot encoding으로 바꿉니다.
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return X_train, X_test, y_train, y_test

def build_cnn_model(num_classes, input_shape):
    model = Sequential()

    # TODO: [지시사항 2번] 지시사항 대로 CNN 모델을 만드세요.
    model.add(layers.Conv2D(16, kernel_size=(3,3), padding="same", activation="relu", input_shape=input_shape))
    model.add(layers.Conv2D(32, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))
    model.add(layers.Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))
    model.add(layers.Conv2D(256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))


    return model

def plot_loss(hist):
    # hist 객체에서 train loss와 valid loss를 불러옵니다.
    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xticks(list(epochs))
    
    # ax를 이용하여 train loss와 valid loss를 plot 합니다..
    ax.plot(epochs, train_loss, marker=".", c="blue", label="Train Loss")
    ax.plot(epochs, val_loss, marker=".", c="red", label="Valid Loss")

    ax.legend(loc="upper right")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    fig.savefig("loss.png")

def plot_accuracy(hist):
    # hist 객체에서 train accuracy와 valid accuracy를 불러옵니다..
    train_acc = hist.history["accuracy"]
    val_acc = hist.history["val_accuracy"]
    epochs = np.arange(1, len(train_acc) + 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xticks(list(epochs))
    # ax를 이용하여 train accuracy와와 valid accuracy와를 plot 합니다.
    ax.plot(epochs, val_acc, marker=".", c="red", label="Valid Accuracy")
    ax.plot(epochs, train_acc, marker=".", c="blue", label="Train Accuracy")

    ax.legend(loc="lower right")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    fig.savefig("accuracy.png")

def get_topk_accuracy(y_test, y_pred, k=1):
    # one-hot encoding으로 이루어진(y_test를 다시 정수 라벨 형식으로 바꿉니다.
    true_labels = np.argmax(y_test, axis=1)

    # y_pred를 확률값이 작은 것에서 큰 순서로 정렬합니다.
    pred_labels = np.argsort(y_pred, axis=1)

    correct = 0
    for true_label, pred_label in zip(true_labels, pred_labels):
        # TODO: [지시사항 3번] 현재 pred_label에서 확률값이 가장 큰 라벨 k개를 가져오세요
        cur_preds = pred_label[-k:]

        if true_label in cur_preds:
            correct += 1

    # TODO: [지시사항 3번] Top-k accuarcy를 구하세요.
    topk_accuracy = correct / len(true_labels)

    return topk_accuracy

def main(model=None, epochs=5):
    # 시드 고정을 위한 코드입니다. 수정하지 마세요!
    tf.random.set_seed(2022)

    X_train, X_test, y_train, y_test = load_cifar10()
    cnn_model = build_cnn_model(len(y_train[0]), X_train[0].shape)
    cnn_model.summary()

    # TODO: [지시사항 4번] 지시사항 대로 모델의 optimizer, loss, metrics을 설정하세요.
    optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    cnn_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # TODO: [지시사항 5번] 지시사항 대로 hyperparameter를 설정하여 모델을 학습하세요.
    hist = cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split= 0.2, shuffle= True, verbose=2)

    # Test 데이터를 적용했을 때 예측 확률을 구합니다.
    y_pred = cnn_model.predict(X_test)
    top1_accuracy = get_topk_accuracy(y_test, y_pred)
    top3_accuracy = get_topk_accuracy(y_test, y_pred, k=3)
    
    print("Top-1 Accuracy: {:.3f}%".format(top1_accuracy * 100))
    print("Top-3 Accuracy: {:.3f}%".format(top3_accuracy * 100))

    # Test accuracy를 구합니다.
    _, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)

    # Tensorflow로 구한 test accuracy와 top1 accuracy는 같아야 합니다.
    # 다만 부동 소수점 처리 문제로 완전히 같은 값이 나오지 않는 경우도 있어서
    # 소수점 셋째 자리까지 반올림하여 비교합니다.
    assert round(test_accuracy, 3) == round(top1_accuracy, 3)

    plot_loss(hist)
    plot_accuracy(hist)

    return optimizer, hist

if __name__ == '__main__':
    main()
    

# 5번문제 > 