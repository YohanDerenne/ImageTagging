import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

import cv2

import os

def training():
    folder = "./Tagging/"

    x_set = []
    y_set = []

    for file in os.listdir(folder):
        if file != "GT":
            x_set.append(cv2.imread(folder + file))
            labels = open(folder + "GT/" + file.split('.')[0] + ".txt").read()
            y_set.append(labels)

    print(x_set)
    print(y_set)



if __name__ == '__main__':
    print('Hello')
    training()