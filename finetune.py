from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from vggface import VGGFace

from sklearn.metrics import log_loss

from one_face_predict import prdict_one_face
from load_face_data import load_face_data

from facetool import FaceTool

def train_face_model(finetune = True):
    #===============custom parameters =============== #

    hidden_dim = 512

    img_width, img_height = 224, 224

    nb_class = 16
    One_Class_Train_MAX = 30
    One_Class_Valid_MAX = 10
    nb_train_samples = nb_class * One_Class_Train_MAX
    nb_validation_samples = nb_class * One_Class_Valid_MAX
    nb_epoch = 10
    batch_size = 16
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'

    save_model_path = './faceDB/face-model.json'
    save_model_h5 = './faceDB/face-model.h5'
    save_face_index = './faceDB/face-index.json'

    # =============== NN =============== #
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))

    # print('----------------After Add finetune layers----------------')
    # for l in vgg_model.layers:
    #     print('Name ', l.name, 'trainable' ,l.trainable)


    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    out = Dense(nb_class, activation='softmax', name='fc8')(x)
    custom_vgg_model = Model(vgg_model.input, out)


    if finetune:
        # print('----------------After Disable Trainable----------------')
        all_layers = custom_vgg_model.layers
        pool5_index = custom_vgg_model.layers.index(custom_vgg_model.get_layer('pool5'))

        for ind, l in enumerate(all_layers):
            if ind <= pool5_index:
                l.trainable = False
        # all_layers[:pool5_index].trainable = False

        
        # for ind, l in enumerate(all_layers):
        #     print('Name ', l.name, 'trainable' ,l.trainable,'index',ind)


    # Train your model as usual.
    # You can Try different optimizers
    # opt = optimizers.SGD(lr=1e-5, decay=1e-6)  #OK
    # adagrad = optimizers.Adagrad( decay=1e-6)
    # opt = optimizers.Adadelta( )

    opt = optimizers.Adam(lr=1e-5, decay=1e-6)
    custom_vgg_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    custom_vgg_model.summary()



    X_train, Y_train, X_valid, Y_valid, Face_Label_Dic = load_face_data('data/')


    ftool = FaceTool()
    ftool.write_json(save_face_index,Face_Label_Dic)

    # Start Fine-tuning
    custom_vgg_model.fit(X_train, Y_train,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                shuffle=True,
                verbose=1,
                validation_data=(X_valid, Y_valid),
                )

    # Make predictions
    predictions_valid = custom_vgg_model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)

    # ===============Save Model===============
    print("Saved model to disk")
    model_json = custom_vgg_model.to_json()
    with open(save_model_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    custom_vgg_model.save_weights(save_model_h5)



    # ===============Test===============
    face_index = prdict_one_face(custom_vgg_model, 'data/test/1.jpg')
    print Face_Label_Dic[face_index]

    face_index = prdict_one_face(custom_vgg_model, 'data/test/2.jpg')
    print Face_Label_Dic[face_index]

    face_index = prdict_one_face(custom_vgg_model, 'data/test/3.jpg')
    print Face_Label_Dic[face_index]


if __name__ == '__main__':   
    train_face_model(False)