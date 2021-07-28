import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam


# Create Autoencoder
def LCA_Net(image_size, gray=False):
    kernel_size = 3  # (3, 3)
    pooling_size = 2  # (2, 2)

    image_shape = np.concatenate((image_size, [1] if gray else [3]))

    model = Sequential()

    # Input layer
    model.add(Input(shape=image_shape))

    #########
    # Encoder
    # Conv
    model.add(Conv2D(50, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(AveragePooling2D(pooling_size, padding='same'))
    model.add(Conv2D(50, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(AveragePooling2D(pooling_size, padding='same'))
    # Dense
    model.add(Dense(10))

    # Decoder
    # Dense
    model.add(Dense(10))
    # Conv
    model.add(Conv2D(50, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(UpSampling2D(pooling_size))
    model.add(Conv2D(50, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(UpSampling2D(pooling_size))

    # Output layer
    # model.add(Conv2D(3, kernel_size=kernel_size, activation='sigmoid', padding='same'))
    model.add(Conv2D(3, kernel_size=kernel_size, activation='relu', padding='same'))

    # Compile & build model
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)   # this settings are default
    model.compile(optimizer=opt, loss='mse')
    # model.compile(optimizer='adam', loss='binary_crossentropy')
    model.build()

    model.summary()

    return model
