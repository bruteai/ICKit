import os
from tensorflow.keras.models import load_model
from argparse import ArgumentParser
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as incept_resnet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
import sys

def predict(model):
    try:
        images=os.listdir(args.path)
    except:
        images=[args.path]
    if args.model == 'InceptionResNetV2':
        preprocess_model=incept_resnet_preprocess
        input_shape=(299,299)
    
    elif args.model == 'MobileNetV2':
        preprocess_model=mobilenet_preprocess
        input_shape=(224,224)
    
    else:
        preprocess_model=efficientnet_preprocess
        input_shape=(224,224)
    
    if args.dataset == 'EMNIST':
        input_shape=(50,50)
        a=list(range(27))
        b=[chr(i) for i in range(65,91)]
        mappings=dict(zip(a,b))
        
    else:
        mappings={0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
    
    for im in images:
        try:
            image = load_img(os.path.join(args.path,im), target_size=input_shape)
        except FileNotFoundError:
            image = load_img(im, target_size=input_shape)

        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_model(image)
        pred=model.predict(image).argmax(axis=-1)[0]
        class_name=mappings[pred]
        print('{} is predicted as {}.'.format(im,class_name))
        
def initialise():
    filename='weights/'+args.model+'_'+args.dataset+'.h5'
    try:
        model=load_model(filename)
        print('Loaded model ',filename)
    except OSError:
        print('No module found for this combination(Model: {},Dataset: {}).\nPlease run train.py to train model first.'.format(args.model,args.dataset))
        sys.exit()
        
    print(model.summary())
    predict(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p',"--path", help="Path to your test dir",required=True)
    parser.add_argument('-d',"--dataset", help="Select your dataset",required=True,choices=['CIFAR10','EMNIST','custom_dataset'])
    parser.add_argument('-m',"--model", help="Select your model",choices=['InceptionResNetV2','MobileNetV2','EfficientNetB7'],default='InceptionResNetV2')
    args = parser.parse_args()
    initialise()    
    
    
#python test.py -p test_folder -d CIFAR10