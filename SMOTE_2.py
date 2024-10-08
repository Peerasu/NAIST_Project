import pandas as pd
import numpy as np
import os
import sys
from shutil import copyfile
import os.path
import cv2
import numpy as np
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array,load_img
from PIL import Image
from sklearn.model_selection import train_test_split
from numpy import load
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

imagegen = ImageDataGenerator()

# load train data from drive
train_generator=imagegen.flow_from_directory('../../../../../mnt/d/peerasu/New/Patch_Train', class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224),seed=42 )
x=np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())])
y=np.concatenate([train_generator.next()[1] for i in range(train_generator.__len__())])
print(x.shape)
print(y.shape)
#Converting  our color images to a vector
X_train=x.reshape(train_generator.__len__(),224*224*3)

#Apply SMOTE method 
sm = SMOTE(random_state=2)
X_smote, y_smote = sm.fit_resample(X_train, y)

#Retrieve the image and save it to drive. Here's an example for a single image
Xsmote_img=X_smote.reshape(585,224,224,3)
pil_img = array_to_img(Xsmote_img[80] * 255)
pil_img.save('/content/gdrive/My Drive/demo.jpg')

#Save all images generated by the SMOTE method to the drive

train_sep_dir = '../../../../../mnt/d/peerasu/New/SMOTE_Patch_Train'

#Create a "testfolder" if it does not exist on the drive
if not os.path.exists(train_sep_dir):
    os.mkdir(train_sep_dir)

#This function return label name 
def get_key(val): 
    for key, value in patho.items(): 
         if val == value: 
  return key 

for i in range(len(Xsmote_img)):
  label=get_key(str(y_smote[i]))  
  if not os.path.exists(train_sep_dir + str(label)):
    os.mkdir(train_sep_dir + str(label))
  pil_img = array_to_img(Xsmote_img[i]* 255)
  pil_img.save(train_sep_dir + str(label) +'/'+ 'smote_'+ str(i) + '.jpg')