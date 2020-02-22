print("[INFO] Importing Libraries")
import matplotlib as plt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# matplotlib inline
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import time   # time1 = time.time(); print('Time taken: {:.1f} seconds'.format(time.time() - time1))
import warnings
import keras
from keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings("ignore")
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import regularizers
from keras import optimizers
from keras.layers import LeakyReLU
from keras.layers import ELU
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from PIL import Image 
import numpy
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import time
from sklearn.metrics import classification_report, confusion_matrix
from keras_applications.resnet import ResNet50
from keras_applications.mobilenet import MobileNet
SEED = 50   # set random seed
print("[INFO] Libraries Imported")

anadelta = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
#adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#leakyrelu = keras.layers.LeakyReLU(alpha=0.3)
#elu = keras.layers.ELU(alpha=1.0)


input_img = Input(shape=(32, 32, 3)) 

#%%
from keras.applications.vgg16 import VGG16



#%%    
def make_model():
    
 print("[INFO] Compiling Model...")

 base_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=2)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
 layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

    
 for layer in base_model.layers:
     layer.trainable = False
  
#     for layer in base_model.layers[30:]:
# 	    layer.trainable = True  
    
 #base_model.summary()
 
 x = layer_dict['block4_conv1'].output
 x= BatchNormalization()(x)
 x = Flatten()(x)
 x = Dense(512, activation='relu')(x)
    #x = Dropout(0.5)(x)
 x = Dense(256, activation='relu')(x)
    #x = Dropout(0.5)(x)
 x = Dense(2, activation='softmax')(x)
 model = Model(input=base_model.input, output=x)
 #model.summary()

 model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 print("[INFO] Model Compiled!")
 return model
  
#%%
  
print("[INFO] loading images from private data...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images('C:\\Users\\User\\spn_myone\\spn_pr')))   # data folder with 2 categorical folders
random.seed(SEED)
random.shuffle(imagePaths)


# loop over the input images
for imagePath in imagePaths:
    # load the image, resize the image to be 32x32 pixels (ignoring aspect ratio), 
    # flatten the 32x32x3=3072 pixel image into a list, and store the image in the data list
    image = cv2.imread(imagePath)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32))/255
    data.append(image)
 
    # extract the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float")
labels = np.array(labels)

print("[INFO] Private data images loaded!")

print("Reshaping data!")

#data = data.reshape(data.shape[0], 32, 32, 1)

print("Data Reshaped to feed into models channels last")

print("Labels formatting")
lb = LabelBinarizer()
labels = lb.fit_transform(labels) 
labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
print("Labels ok!")

print("[INFO] loading images from dataset for test...")
data2 = []
labels2 = []

# grab the image paths and randomly shuffle them
imagePaths2 = sorted(list(paths.list_images('C:\\Users\\User\\spn_myone\\lidc_sel')))   # data folder with 2 categorical folders
random.seed(SEED)
random.shuffle(imagePaths2)


# loop over the input images
for imagePath2 in imagePaths2:
    # load the image, resize the image to be 32x32 pixels (ignoring aspect ratio), 
    # flatten the 32x32x3=3072 pixel image into a list, and store the image in the data list
    image1 = cv2.imread(imagePath2)
    #image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (32, 32))/255
    data2.append(image1)
 
    # extract the class label from the image path and update the labels list
    label2 = imagePath2.split(os.path.sep)[-2]
    labels2.append(label2)


# scale the raw pixel intensities to the range [0, 1]
data2 = np.array(data2, dtype="float")
print("[INFO] Test data images loaded!")


lb = LabelBinarizer()
labels2 = lb.fit_transform(labels2) 
labels2 = keras.utils.to_categorical(labels2, num_classes=2, dtype='float32')
#data2 = data2.reshape(data2.shape[0], 32, 32, 1)



#%%
time1 = time.time() #initiate time counter
n_split=10 #10fold cross validation
scores = [] #here every fold accuracy will be kept
predictions_all = np.empty(0) # here, every fold predictions will be kept
predictions_all_num = np.empty([0,2]) # here, every fold predictions will be kept
test_labels = np.empty(0) #here, every fold labels are kept
name2 = 5000 #name initiator for the incorrectly classified insatnces
conf_final = np.array([[0,0],[0,0]]) #initialization of the overall confusion matrix

for train_index,test_index in KFold(n_split).split(data): #data split
    trainX,testX=data[train_index],data[test_index]
    trainY,testY=labels[train_index],labels[test_index]
    
    trainX = np.concatenate([trainX, data2])
    trainY =  np.concatenate([trainY, labels2])
    
    
    
    model3 = make_model() #in every iteration we retrain the model from the start and not from where it stopped
   
    
    model3.fit(trainX, trainY,epochs=5, batch_size=32)
    
    #aug = ImageDataGenerator(rotation_range=10, horizontal_flip=True, vertical_flip=True, fill_mode = 'nearest')
    #aug.fit(trainX)
    #model3.fit_generator(aug.flow(trainX, trainY,batch_size=32), epochs=100, steps_per_epoch=len(trainX)//32)
    
    
    
    score = model3.evaluate(testX,testY)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    print('Model evaluation ',model3.evaluate(testX,testY))
   
   
    #predict = model.predict_classes(testX)
    predict = model3.predict(testX) #for def models functional api
    predict_num = predict
    predict = predict.argmax(axis=-1) #for def models functional api
    conf = confusion_matrix(testY2, predict) #get the fold conf matrix
    conf_final = conf + conf_final #sum it with the previous conf matrix
    name2 = name2 + 1
   
    predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
    predictions_all_num = np.concatenate([predictions_all_num, predict_num])
    test_labels = np.concatenate ([test_labels, testY2]) #merge the two np arrays of labels

# BELOW IS A CODE TO SAVE THE INCORRECTLY CLASSIFIED INSTANCES    
    testX_one = testX[:, :, :, 0] #remove the last column from the image array
    for i in range (len(testX)): #for every image in testX
            if testY2[i] != predict[i]: #if the image is incorrectly classified
                    im = Image.fromarray(testX_one[i]) #take the array and convert to image
                    im2 = numpy.array(im, dtype=float)*255 # do it numpy array again an resize the pixel values
                    im3 = Image.fromarray(im2) # convert back to image 
                    im3 = im3.convert('RGB') # convert to 3channel
                    name = i #give the img a unique name
                    name = 'C:\\Users\\User\\spn_myone\\inc\\in' + str(name) +str(name2)
                    im3.save(name + '.jpeg') #save to folder
  

scores = np.asarray(scores)
final_score = np.mean(scores)




print("[INFO] Results Obtained!")
print('Time taken: {:.1f} seconds'.format(time.time() - time1)) 








#%%

test_labels_2 = keras.utils.to_categorical(test_labels, num_classes=2, dtype='float32')

from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(test_labels_2[:, i], predictions_all_num[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_2.ravel(), predictions_all_num.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(2):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 2

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(2), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()









#%%
prediction2 = model.predict_classes(data3)
from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
print(confusion_matrix(labels2, prediction2))

print('Classification Report')
target_names = ['Benign', 'Malignant']
print(classification_report(labels2, prediction2, target_names=target_names)) 
  
  

