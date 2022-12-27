import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import os
from os import listdir

for _data_name in ['TestA', 'TestB', 'Val']: 

    path = '/content/drive/MyDrive/BTP/{}/images/'.format(_data_name)
    i = 0
    print(_data_name)
    for f in os.listdir(path):
        if i == 5:
         break
        ipath = path + f
        mpath = '/content/drive/MyDrive/BTP/{}/'.format(_data_name) + 'masks/' + f
        ppath = '/content/drive/MyDrive/BTP/' + 'result_map/PolypPVT/{}/'.format(_data_name) + f
        img1 = cv2.imread(ipath)
        half1 = cv2.resize(img1, (0, 0), fx = 0.4, fy = 0.4)
        img2 = cv2.imread(mpath)
        half2 = cv2.resize(img2, (0, 0), fx = 0.4, fy = 0.4)
        img3 = cv2.imread(ppath)
        half3 = cv2.resize(img3, (0, 0), fx = 0.4, fy = 0.4)
        print("              Image                                   Ground Truth                              Predicted")
        Hori = np.concatenate((half1, half2, half3), axis=1)
        cv2.imshow(Hori)
        i=i+1
