import urllib.request
import time
import numpy as np
import cv2

def cam():
    url='http://192.168.0.102:8080/shot.jpg'


    # Use urllib to get the image from the IP camera
    imgResp = urllib.request.urlopen(url)
    
    # Numpy to convert into a array
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    # Finally decode the array to OpenCV usable format ;) 
    image1 = cv2.imdecode(imgNp,-1)
    return image1