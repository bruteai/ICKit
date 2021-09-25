# Train the model
from argparse import ArgumentParser
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import datasets, layers, models, losses, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization,Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD

from Models.InceptionResNet import get_incep_resnet_model
from Models.EfficientNet import get_EffcientNet_model
from Models.MobileNet import get_MobileNet_model
from preprocessing import CIFAR10 as CIFAR10, EMNIST

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

def train_model(input_shape,num_classes):
    if args.model == 'InceptionResNetV2':
        model=get_incep_resnet_model(input_shape,num_classes)

    if args.model == 'EfficientNetB7':
        model=get_EffcientNet_model(input_shape,num_classes)
        
    if args.model == 'MobileNetV2':
        model=get_MobileNet_model(input_shape,num_classes)
    
    lr = 1e-3
    sgd = SGD(lr = lr, momentum = 0.9, nesterov = True)
    model.compile(optimizer= sgd, loss= 'categorical_crossentropy', metrics= ['accuracy'])
    return model

def fit_generator(train_generator,valid_generator,input_shape):
    #model.compile(optimizer='adam',loss ="categorical_crossentropy",metrics='accuracy')
    step_size_valid = valid_generator.n//valid_generator.batch_size
    step_size_train = train_generator.n//train_generator.batch_size
    model.fit(train_generator,
                  epochs=30,
                  steps_per_epoch=step_size_train, 
                  batch_size=args.batch_size,
                  validation_data=valid_generator,
                  validation_steps=step_size_valid,
                  verbose=1
                  )
        
def fit_model(model,X_train, y_train, X_test, y_test,input_shape):
    step_size_train = X_train.shape[0]//args.batch_size
    step_size_valid = X_test.shape[0]//args.batch_size
    print(X_train.shape)
    print(y_train)
    print('--------------------------------')
    print(X_test.shape)
    print(y_test)
    model.fit(X_train,
               y_train,
               epochs=30,
               steps_per_epoch=step_size_train, 
               batch_size=args.batch_size,
               validation_data=(X_test,y_test),
               validation_steps=step_size_valid,
               verbose=1
               )
    
def get_input_shape():
    if args.model == 'InceptionResNetV2':
        return (299, 299, 3)
    
    if args.model == 'MobileNetV2':
        return (224, 224, 3)
    
    if args.model == 'EfficientNetB7':
        return (224, 224, 3)
    
def getdata():
    if args.dataset == 'CIFAR10':
        input_shape=get_input_shape()
        cifar=CIFAR10(args.see_samples)
        df=cifar.preprocess()
        train_generator,valid_generator=cifar.load_data(df,args.model,input_shape,args.batch_size)
        num_classes=10
        model=train_model(input_shape,num_classes)
        fit_generator(model,train_generator,valid_generator,input_shape)
        
    elif args.dataset == 'EMNIST':
        emnist=EMNIST()
        X_train, y_train, X_test, y_test=emnist.load_data()
        X_train, y_train, X_test, y_test=emnist.preprocess_data(X_train, y_train, X_test, y_test)
        n,rows,cols,channels=X_train.shape
        input_shape=(rows,cols,channels)
        if args.see_samples:
            emnist.visualise(X_train,y_train)
        num_classes=26
        model=train_model(input_shape,num_classes)
        fit_model(model,X_train, y_train, X_test, y_test,input_shape)
        


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--see_samples", help="See a few sample images from the dataset",default=False)
    parser.add_argument('-d',"--dataset", help="Select your dataset",required=True,choices=['CIFAR10','EMNIST','custom_dataset'])
    parser.add_argument('-m',"--model", help="Select your model",choices=['InceptionResNetV2','MobileNetV2','EfficientNetB7'],default='Resnet50')
    parser.add_argument('-bsize',"--batch_size", help="increase output verbosity",default=32, type=int)
    parser.add_argument("--visualise", help="Visualise Training and Testing accuracy",default=False)
    args = parser.parse_args()
    getdata()    
    

#python train.py -d EMNIST -m EfficientNetB7 --see_samples True
#python train.py -d CIFAR10 -m Resnet50
