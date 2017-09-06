# vgg-face-keras

Thank for the https://github.com/rcmalli/keras-vggface

This repo is modified form the rcmalli repo: [rcmalli/keras-vggface](https://github.com/rcmalli/keras-vggface)


## Download face data from this link

1. [Face Data](https://drive.google.com/open?id=0BysSXLPvHi7DOHlkMTZNa2R3NlU)
2. Extract to data/faces/, data/train/, data/validation, data/test

## Download haarcascade_frontalface_default.xml
1. Please download cascade XML file from this repo : https://github.com/shantnu/FaceDetect.git

2. copy to this directory
faceDB/haarcascade_frontalface_default.xml



## Run
```
python finetune.py
```

## Try Different Parameter


You can try modify following parameter 
```
~fintune.py~
nb_class = 16  
One_Class_Train_MAX = 30
One_Class_Valid_MAX = 10
nb_epoch = 10
```