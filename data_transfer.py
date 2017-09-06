import numpy as np


def data_transfer(x):
    # # TF order aka 'channel-last'
    x = x[:, :, :, ::-1]
    # TH order aka 'channel-first'
    # x = x[:, ::-1, :, :]
    # Zero-center by mean pixel
    x[:, 0, :, :] -= 93.5940
    x[:, 1, :, :] -= 104.7624
    x[:, 2, :, :] -= 129.1863

    # # Switch RGB to BGR order 
    # x = x[:, :, :, ::-1]  

    # # Subtract ImageNet mean pixel 
    # x[:, :, :, 0] -= 103.939
    # x[:, :, :, 1] -= 116.779
    # x[:, :, :, 2] -= 123.68


    return x