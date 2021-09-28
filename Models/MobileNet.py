# Mobilenet Model - Fast
from tensorflow.keras.applications import MobileNetV2
from keras.layers import Dense,GlobalAveragePooling2D, Conv2D, Reshape, Activation, Dropout
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization,Activation, Input,GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model

def get_MobileNet_model(input_shape,num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape,classes=num_classes)
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

def old():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32,32,3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,1280))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(10,(1,1),padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((10,) )(x)
    
    model = Model(inputs = base_model.input, outputs = output)
    
    for layer in model.layers:
        layer.trainable = True
    
    return model
