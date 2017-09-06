from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from facetool import FaceTool
import cv2 
#----Init FaceTool-----#
ftool = FaceTool()

def prdict_one_face(i_model, img_path = '', img = None):
    if img_path != '':
        # img = image.load_img(img_path, target_size=(224, 224))
        img = cv2.imread(img_path)

    height, width, channels = img.shape
    if height != 224 or width != 224:
        img = cv2.resize(img, (224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # For Tensorflow 
    # Switch RGB to BGR order 
    x = x[:, :, :, ::-1]  

    # Subtract ImageNet mean pixel 
    # x[:, :, :, 0] -= 103.939
    # x[:, :, :, 1] -= 116.779
    # x[:, :, :, 2] -= 123.68
    x[:, 0, :, :] -= 93.5940
    x[:, 1, :, :] -= 104.7624
    x[:, 2, :, :] -= 129.1863

    preds = i_model.predict(x)

    # print all preds
    # print('preds:',preds)  

    max_pred = np.max(preds[0])
    max_arg = np.argmax(preds[0])
    print('Predicted:', max_arg,  '{:2f}%'.format( max_pred * 100. ))
    
    if max_pred > 0.8:
        return np.argmax(preds[0])
    else:   
        return -1 

if __name__ == '__main__':
    """ Load Model """
    # load json and create model
    json_file = open('faceDB/face-model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("faceDB/face-model.h5")
    print("Loaded model from disk")
    
    face_index = prdict_one_face(loaded_model, img_path = 'data/test/1.jpg')
    print ftool.get_face_label(face_index)

    face_index = prdict_one_face(loaded_model, img_path = 'data/test/2.jpg')
    print ftool.get_face_label(face_index)

    # img = image.load_img('data/test/test.jpg')
    
    img = cv2.imread('data/test/test.jpg')
    face_index = prdict_one_face(loaded_model, img = img )
    print ftool.get_face_label(face_index)

