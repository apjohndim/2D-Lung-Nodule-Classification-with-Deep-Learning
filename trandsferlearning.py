import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LocallyConnected2D, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import regularizers
import time
from keras import optimizers
from keras.layers import LeakyReLU
from keras.layers import ELU
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from PIL import Image 
import numpy

print("[INFO] Libraries Imported!")
anadelta = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
leakyrelu = keras.layers.LeakyReLU(alpha=0.3)
elu = keras.layers.ELU(alpha=1.0)


#%%

loaded_model = load_model('threeway.h5')


#%%

for layer in loaded_model.layers[:-2]:
    layer.trainable = False
    
loaded_model.summary()

#%%
loaded_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#%%

time1 = time.time()
H = loaded_model.fit_generator(aug.flow(trainX, trainY,batch_size=32), epochs=200, steps_per_epoch=len(trainX)//32)
print('Time taken: {:.1f} seconds'.format(time.time() - time1)) 

#%%

prediction = loaded_model.predict(testX)
prediction_class = prediction.argmax(axis=-1)
from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
print(confusion_matrix(testY2, prediction_class))
print('Classification Report')
target_names = ['Benign', 'Malignant']
print(classification_report(testY2, prediction_class, target_names=target_names))