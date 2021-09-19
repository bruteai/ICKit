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
class CIFAR10old:
    def __init__(self,samples,path='dataset/old'):
        self.dataset_path=path
        
    def add_test_to_train(self):
        train_path=os.path.join(self.dataset_path,'train')
        test_path=os.path.join(self.dataset_path,'test')
        class_name=os.listdir(test_path)
        for name in class_name:
            for root, dirs, files in os.walk(os.path.join(test_path,name)):
                train_folder=root.replace('test','train')
                print(train_folder)
                print('root',root)
                print(len(files))
                files_to_move=[os.path.join(root,f) for f in files[:-5]]
                print(len(files_to_move))
                #for name in files:
                #    r.append(os.path.join(root, name))
                #    return r
                
    def load_data(self):
        print('Loading data....')
        train_path=os.path.join(self.dataset_path,'train')
        valid_path=os.path.join(self.dataset_path,'test')
        train_datagen = ImageDataGenerator(rescale=1./255,
                                           #shear_range=0,
                                           #zoom_range=0,
                                           #rotation_range=30,
                                           vertical_flip=False,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1)

        test_datagen = ImageDataGenerator(rescale=1./255)  

        train_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=(32, 32),
                batch_size=32,
                class_mode='categorical',
                shuffle=False)
        
        validation_generator = test_datagen.flow_from_directory(
                valid_path,
                target_size=(32, 32),
                batch_size=32,
                class_mode='categorical',

                shuffle=False)
        
        return train_generator,validation_generator

         
        
       
class CIFAR10:
    def __init__(self,samples,path='dataset/CIFAR10'):
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
        #imgs_out = np.zeros((100,rows,cols,channels))
        #X_train = np.expand_dims(X_train, axis=-1)
        #X_test = np.expand_dims(X_test, axis=-1)
        #print(X_train.shape)
        print('Preprocessing data....')
        X_train=X_train[:80000,:,:]
        y_train=y_train[:80000]
        n,rows,cols=X_train.shape
        X_train = grayscale_to_rgb(tf.expand_dims(X_train, axis=3),name=None)
        #X_train = np.expand_dims(X_train, axis=-1).repeat(3, -1)
        X_test = grayscale_to_rgb(tf.expand_dims(X_test, axis=3),name=None)
        print('done')
        X_train_out = resize(X_train, [100,100])
        X_test_out = resize(X_test, [100,100])
        #X_train_out = np.zeros((X_train.shape[0],32,32,3))
        #for n,i in enumerate(X_train):
        #    X_train_out[n,:,:,:] = resize(X_train[n,:,:,:], X_train_out.shape[1:], anti_aliasing=True)
            
        #X_test_out = np.zeros((X_test.shape[0],32,32,3))
        #for n,i in enumerate(X_test):
        #    X_test_out[n,:,:,:] = resize(X_test[n,:,:,:], X_test_out.shape[1:], anti_aliasing=True)
        #X_train = X_train.reshape(X_train.shape[0], rows, cols, )
        #X_test = X_test.reshape(X_test.shape[0], rows, cols, 1)
        X_train_out /= 255
        X_test_out /= 255
        #X_train_out, y_train = shuffle(X_train_out, y_train)
        #X_test_out, y_test = shuffle(X_test_out, y_test)
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
        
    def visualise(self,X_train,y_train):
        fig, ax = plt.subplots(5, 5, figsize=(10,10))
        for i, axi in enumerate(ax.flat):
            axi.imshow(X_train[i])
            axi.set(xticks=[], yticks=[],xlabel=chr(y_train[i]+64))
        
        plt.show()
                
    
if __name__ == '__main__':
    CatVsDog()
    
    