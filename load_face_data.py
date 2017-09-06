import cv2, numpy as np
from keras.utils import np_utils
import os
from data_transfer import data_transfer
from shutil import copyfile, rmtree

def load_face_data(root_dir):
    TrainDir = root_dir + 'train/'
    ValidationDir = root_dir + 'validation/'

    num_classes = 16
    img_rows, img_cols = 224, 224
    One_Class_Train_MAX = 30
    One_Class_Valid_MAX = 10
    num_train_samples = num_classes * One_Class_Train_MAX
    num_valid_samples = num_classes * One_Class_Valid_MAX

    Face_Label_Dic = {}

    def make_train_class_ary(target_dir, num_train_samples, class_max):
        x_train = np.zeros((num_train_samples, img_rows, img_cols, 3), dtype=np.float32)
        y_train = np.zeros((num_train_samples), dtype='uint8')


        class_count = 0
        train_count = 0
        for subdir, dirs, files in os.walk(target_dir):
            # print('subdir = {0}, dirs = {1}, files = {2}'.format(subdir, dirs, files)) 
            if subdir == target_dir:
                continue

            # print('Load Img from dir: ' + subdir)
            # print('class_count={0}, train_count={1}'.format(class_count, train_count))
            
            basename = os.path.basename(subdir)
            Face_Label_Dic[class_count] = basename

            this_class_train_count = 0
            for f in files:
                # print('Load Img : ' + subdir + '/' + f)
                im = cv2.imread(subdir + '/' + f)
                im_np_ary = np.asarray(cv2.resize(im, (img_rows,img_cols)).astype(np.float32)   )

                x_train[train_count, : , :, :] = im_np_ary
                y_train[train_count] = class_count
                this_class_train_count += 1
                train_count += 1

                if this_class_train_count >= class_max:
                    break

            class_count += 1

            if class_count >= num_classes:
                break

        return x_train, y_train


    X_Train, Y_Train = make_train_class_ary(TrainDir, num_train_samples, One_Class_Train_MAX)
    X_Valid, Y_Valid = make_train_class_ary(ValidationDir, num_valid_samples, One_Class_Valid_MAX)

    # print('Y_Train')
    # print(Y_Train)
    # print('Y_Valid')
    # print(Y_Valid)

    # Transform targets to keras compatible format
    Y_Train = np_utils.to_categorical(Y_Train[:num_train_samples], num_classes)
    Y_Valid = np_utils.to_categorical(Y_Valid[:num_valid_samples], num_classes)

    # print('After Y_Train')
    # print(Y_Train)

    # print('After Y_Valid')
    # print(Y_Valid)

    # Switch RGB to BGR order 
    # X_Train = X_Train[:, :, :, ::-1]  

    # # Subtract ImageNet mean pixel 
    # X_Train[:, :, :, 0] -= 103.939
    # X_Train[:, :, :, 1] -= 116.779
    # X_Train[:, :, :, 2] -= 123.68

    # # Switch RGB to BGR order 
    # X_Valid = X_Valid[:, :, :, ::-1]  

    # # Subtract ImageNet mean pixel 
    # X_Valid[:, :, :, 0] -= 103.939
    # X_Valid[:, :, :, 1] -= 116.779
    # X_Valid[:, :, :, 2] -= 123.68

    X_Train = data_transfer(X_Train)

    X_Valid = data_transfer(X_Valid)


    return X_Train, Y_Train, X_Valid, Y_Valid, Face_Label_Dic
    

def test_load_face_data():
    if not os.path.isdir('dump_ary_img'):
        os.mkdir('dump_ary_img')


    to_train_dir = 'dump_ary_img/train'
    if os.path.isdir(to_train_dir):
        rmtree(to_train_dir)
    os.mkdir(to_train_dir)

    to_valid_dir = 'dump_ary_img/valid'
    if os.path.isdir(to_valid_dir):
        rmtree(to_valid_dir)
    os.mkdir(to_valid_dir)

    X_Train, Y_Train, X_Valid, Y_Valid = load_face_data('data/')
     

    # return
    print('len(X_Train) = ', len(X_Train))
    print('len(X_Valid) = ', len(X_Valid))

    for i in range(len(X_Train)):
        save_path = to_train_dir + '/' + str(i) + '.jpg'
        print('Save to '+ save_path)
        cv2.imwrite(save_path, X_Train[i,:,:,:])

    for i in range(len(X_Valid)):
        save_path = to_valid_dir + '/' + str(i) + '.jpg'
        print('Save to '+ save_path)
        cv2.imwrite(save_path, X_Valid[i,:,:,:])

if __name__ == '__main__':   
    test_load_face_data()
