import numpy as np
from functools import reduce
import pandas as pd
import keras
from keras.layers import Dense, Input, BatchNormalization, Concatenate, GlobalAveragePooling2D, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model, load_model

from data import *

def make_model():
    inp_img_1 = Input(shape=(75, 75, 2), name='input1_img')
    inp_img_2 = Input(shape=(75, 75, 2), name='input2_img')

    inp_angle = Input(shape=(1,), name='input_angle')
    
    C1, C2, C3 = 32, 64, 30
    
    img_flow_tpl1 = [BatchNormalization(),
                     Conv2D(C1, (3, 3), activation='relu', padding='valid'),
                     Conv2D(C1, (3, 3), activation='relu', padding='valid'),
                     MaxPooling2D((2, 2), strides=(2, 2)),
                     Dropout(DROPOUT_RATE),

                     Conv2D(C2, (3, 3), activation='relu', padding='valid'),
                     Conv2D(C2, (3, 3), activation='relu', padding='valid'),
                     MaxPooling2D((2, 2), strides=(2, 2)),
                     Dropout(DROPOUT_RATE),
                     
                     Flatten()
                     ]

    angle_flow_tpl1 = [BatchNormalization(),
                      Dense(16, activation='relu')
                     ]
    
    img_flow_tpl2 = [BatchNormalization(),
                     Conv2D(C1, (3, 3), activation='relu', padding='valid'),
                     Conv2D(C1, (3, 3), activation='relu', padding='valid'),
                     MaxPooling2D((2, 2), strides=(2, 2)),
                     Dropout(DROPOUT_RATE),

                     Conv2D(C2, (3, 3), activation='relu', padding='valid'),
                     Conv2D(C2, (3, 3), activation='relu', padding='valid'),
                     MaxPooling2D((2, 2), strides=(2, 2)),
                     Dropout(DROPOUT_RATE),
                     
                     Flatten()
                     ]
                     
    angle_flow_tpl2 = [BatchNormalization(),
                      Dense(16, activation='relu')
                     ]

    img_1_flow = [inp_img_1] + img_flow_tpl1
    img_2_flow = [inp_img_2] + img_flow_tpl2

    x1 = reduce(lambda x, y: y(x), img_1_flow)
    x2 = reduce(lambda x, y: y(x), img_2_flow)

    angle_1_flow = [inp_angle] + angle_flow_tpl1
    angle_2_flow = [inp_angle] + angle_flow_tpl2


    x1_angle = reduce(lambda x, y: y(x), angle_1_flow)
    x2_angle = reduce(lambda x, y: y(x), angle_2_flow)



    x1 = Concatenate(name='features1')([x1, x1_angle])
    x2 = Concatenate(name='features2')([x2, x2_angle])

    x = Concatenate()([x1, x2])
    x = Dense(50, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(50, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model([inp_img_1, inp_img_2, inp_angle], out)

    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])

    return model
	
DROPOUT_RATE = 0.3
N_FOLDS = 1

def main():
    train_X, angles, train_y = prepare_data_train()
    for f in range(N_FOLDS):
        tr, val  = split_data(train_X, angles, train_y, f)

        tr_x, val_x = tr[:-1], val[:-1]
        tr_y, val_y = tr[-1], val[-1]

        model = make_model()
        model.summary()
        model.fit(tr_x, tr_y,
          batch_size=24,
          epochs=2,
          verbose=1,
          validation_data=(val_x, val_y))

        model.save('net' + str(f))

if __name__ == '__main__':
    main()
        
