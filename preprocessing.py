from argparse import ArgumentParser
import os
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as incept_resnet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from emnist import list_datasets, extract_training_samples, extract_test_samples
import matplotlib.pyplot as plt
#from skimage.transform import resize
import numpy as np
from tensorflow.image import grayscale_to_rgb,resize 
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
         
class CIFAR10:
    def __init__(self,path='dataset/CIFAR10'):
        self.dataset_path=path
        
    def preprocess(self):
        print('Preprocessing CIFAR data ....\n')
        df=pd.read_csv('dataset/CIFAR10/trainLabels.csv')
        df['id']=df['id'].apply(lambda x: str(x)+'.png')
        print('Preprocessing Complete!!!\n')
        return df
    
    def load_data(self,df,model,input_shape,batch_size):
        if model == 'InceptionResNetV2':
            preprocess_model=incept_resnet_preprocess
        
        if model == 'MobileNetV2':
            preprocess_model=mobilenet_preprocess
        
        if model == 'EfficientNetB7':
            preprocess_model=efficientnet_preprocess
        
        t1,t2,_=input_shape
        input_shape1=(t1,t2)
        print(input_shape1)
        print('Loading data....')        
        path=os.path.join(self.dataset_path,'data')
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_model, 
                                           rotation_range=2, 
                                           horizontal_flip=True,
                                           zoom_range=.1,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           validation_split=0.2)


        train_generator = train_datagen.flow_from_dataframe(
                dataframe=df,
                directory=path,
                x_col="id",
                y_col="label",
                subset="training",
                batch_size=batch_size,
                seed=42,
                shuffle=True,
                class_mode="categorical",
                target_size=input_shape1)
        
        validation_generator = train_datagen.flow_from_dataframe(
                dataframe=df,
                directory=path,
                x_col="id",
                y_col="label",
                subset="validation",
                batch_size=batch_size,
                seed=42,
                shuffle=True,
                class_mode="categorical",
                target_size=input_shape1)
        print('Data Successfully Loaded!!!\n')
        return train_generator,validation_generator
        
class EMNIST():
    def __init__(self):
        pass
    
    def preprocess_data(self,X_train, y_train, X_test, y_test):
        print('Preprocessing data....')
        X_train=X_train[:50000,:,:]
        y_train=y_train[:50000]
        n,rows,cols=X_train.shape
        X_train = grayscale_to_rgb(tf.expand_dims(X_train, axis=3),name=None)
        X_test = grayscale_to_rgb(tf.expand_dims(X_test, axis=3),name=None)
        X_train_out = resize(X_train, [50,50])
        X_test_out = resize(X_test, [50,50])
        X_train_out=np.divide(X_train_out,255.0)
        X_test_out=np.divide(X_test_out,255.0)
        y_train = to_categorical(y_train, 26)
        y_test = to_categorical(y_test, 26)
        print('Preprocessing Finished!!!\n')
        return X_train_out, y_train, X_test_out, y_test

        
    def load_data(self):
        print('Loading data ....')
        X_train, y_train = extract_training_samples('letters')
        y_train = y_train-1
        X_test, y_test = extract_test_samples('letters')
        y_test=y_test-1
        print('Data Successfully Loaded!!!\n')
        return X_train, y_train, X_test, y_test
        
                
    
if __name__ == '__main__':
    CatVsDog()
    
    