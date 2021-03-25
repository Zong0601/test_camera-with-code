#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.compat.v1.disable_eager_execution()


# In[2]:


from __future__ import print_function,division
from tensorflow import keras
import scipy
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import numpy as np
import os
import time
import cv2
from glob import glob
from PIL import Image
from keras.models import load_model
from keras_contrib.layers import InstanceNormalization

np.random.seed(10)


# In[3]:


def imread(path):          
    return scipy.misc.imread(path, mode='RGB').astype(np.float)


# In[4]:


model = load_model('C:/Users/user/Desktop/GAN_TEST/model/ukiyoe256x256_2000/G_BA_ukiyoe256x256.h5',custom_objects={'InstanceNormalization':InstanceNormalization},compile=False)


# In[11]:


cap = cv2.VideoCapture(0)  # 創建一個 VideoCapture 

cap.set(3, 640)            # cap.set（）設置攝像頭參數：  3:寬   4:高
cap.set(4, 480)            # opencv-python 版本太低會沒辦法調整長寬比 


while(cap.isOpened()):  # 循環讀取每一貞數     # cap.isOpened() 檢查攝影機是否有啟動

    ret, frame = cap.read()
    cv2.imshow("camera", frame)    
    
    img = scipy.misc.imresize(frame,(256,256))
    imgs = np.array(img)/127.5 - 1.
    imgs = np.expand_dims(imgs,0)
    #imgs.shape
    
    #start =time.clock()
    x = model.predict(imgs)
    #end = time.clock()
    #print('Running time: %s Sec'%(end-start))
    
    x = (x+1)/2
    x=x[0,:,:]
    #x.shape
    x1 = scipy.misc.imresize(x,(480,640))
    cv2.imshow("camera2", x)
    cv2.imshow("camera3", x1)
    
    k = cv2.waitKey(1) & 0xFF  # 每帧數據延遲 1ms，延遲不能為 0，否則讀取的结果会是静態貞

    if k == ord('s'):  # 檢測到按键 ‘s’，保存圖片
        #print(cap.get(3))
        #print(cap.get(4))
        #保存一帧图片q
        cv2.imwrite('1.jpg', frame)
         
    elif k == ord('q'):  # 檢測到按键 ‘q’，退出
        break

cap.release()
cv2.destroyAllWindows()    


# In[ ]:





# In[ ]:





# In[ ]:




