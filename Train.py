import keras
from scipy.io import loadmat
import numpy as np
import math
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.utils.data_utils import Sequence
from keras import optimizers
from keras import initializers

# define constrans
BATCH_SIZE = 20

# normalize function
def normalize(mat):
    mean = np.mean(mat, axis=1).reshape(-1, 1)
    std = np.std(mat, axis=1).reshape(-1, 1)
    return (mat - mean) / (std + 0.0000001)


# data batch generator
class DataGenerator(Sequence):
    def __init__(self, data_path, label_file, batch_size):
        self.data_path = data_path
        self.label_file = label_file
        self.batch_size = batch_size
        self.labels = pd.read_csv(self.label_file)
        self.size = len(self.labels)
        self.list_IDs = self.labels.index.tolist()
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def __getitem__(self, idx):
        batch_indeces = self.list_IDs[idx * self.batch_size:(idx + 1) * self.batch_size]
        paths = self.labels.iloc[batch_indeces]['path']
        batch_mat = np.array([normalize(loadmat(path)['data'].transpose()) for path in paths])
        batch_classes = np.array(self.labels.iloc[batch_indeces][['if_normal', 'if_abnormal']])
        return batch_mat, batch_classes

    def on_epoch_end(self):
        np.random.shuffle(self.list_IDs)

# create train dataset generator
train_dataset = DataGenerator(data_path='.\preliminary_cl\TRAIN', label_file='trainset_annotation.csv', batch_size=BATCH_SIZE)
val_dataset = DataGenerator(data_path='.\preliminary_cl\TRAIN', label_file='valset_annotation.csv', batch_size=BATCH_SIZE)


model = Sequential()
model.add(Conv1D(16, 16, strides=1, activation='relu', input_shape=(5000, 12), kernel_initializer=initializers.he_normal(seed=3)))
model.add(Conv1D(16, 16, strides=1, activation='relu', padding="same", kernel_initializer=initializers.he_normal(seed=3)))
model.add(MaxPooling1D(2))
model.add(Conv1D(32, 8, strides=1, activation='relu', padding="same", kernel_initializer=initializers.he_normal(seed=3)))
model.add(Conv1D(32, 8, strides=1, activation='relu', padding="same", kernel_initializer=initializers.he_normal(seed=3)))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 4, strides=1, activation='relu', padding="same", kernel_initializer=initializers.he_normal(seed=3)))
model.add(Conv1D(64, 4, strides=1, activation='relu', padding="same", kernel_initializer=initializers.he_normal(seed=3)))
model.add(Conv1D(64, 4, strides=1, activation='relu', padding="same", kernel_initializer=initializers.he_normal(seed=3)))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 2, strides=1, activation='relu', padding="same", kernel_initializer=initializers.he_normal(seed=3)))
model.add(Conv1D(128, 2, strides=1, activation='relu', padding="same", kernel_initializer=initializers.he_normal(seed=3)))
model.add(Conv1D(128, 2, strides=1, activation='relu', padding="same", kernel_initializer=initializers.he_normal(seed=3)))
model.add(MaxPooling1D(2))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.6))
model.add(Dense(2, activation='softmax'))

print(model.summary())

#train
ckpt = keras.callbacks.ModelCheckpoint(
       filepath='best_model.{epoch:02d}-{val_accuracy:.2f}.h5',
       monitor='val_accuracy',
       save_best_only=True, verbose=1)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(
        generator=train_dataset,
        epochs=500,
        initial_epoch=0,
        validation_data=val_dataset,
        callbacks=[ckpt],
        verbose=2
        )
