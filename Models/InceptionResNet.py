from tensorflow.keras.applications import InceptionResNetV2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization,Activation,Input,GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model

def get_incep_resnet_model(input_shape,num_classes):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape,classes=num_classes)
    inputs = Input(shape=input_shape)
    
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x) # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dropout(0.25)(x) # to avoid overfitting
    x = Dense(512, activation='relu')(x) # dense layer 2
    #x = tf.keras.layers.BatchNormalization()(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.layers[1].trainable = False
    return model

    

def old(input_shape):
    base_model = InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = input_shape)
    model_2=Sequential()
    model_2.add(base_model)
    model_2.add(Flatten())
    model_2.add(Dense(4000,activation=('relu'),input_dim=512))
    model_2.add(Dense(2000,activation=('relu'))) 
    model_2.add(Dropout(.4))#Adding a dropout layer that will randomly drop 40% of the weights
    model_2.add(Dense(2000,activation=('relu'))) 
    model_2.add(Dropout(.3))#Adding a dropout layer that will randomly drop 30% of the weights
    model_2.add(Dense(500,activation=('relu')))
    model_2.add(Dropout(.2))#Adding a dropout layer that will randomly drop 20% of the weights
    model_2.add(Dense(10,activation=('softmax'))) #This is the classification layer
    model_2.summary()
    return model_2

