import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class CNN:
    def __init__(self, input_shape=(28, 51, 1)):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = CNN().get_model()
    model.summary()