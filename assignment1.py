import cv2
import numpy as np
import matplotlib.pyplot as plt 
import util as utl

def sequentialChannelEnhancement(img):
  out = np.zeros_like(img, dtype='uint8')
  
  m,n,space = img.shape
  
  for i in range (space):
    out[:,:,i] = utl.gammaTrans(img[:,:,i], 0.4)
  
  return out
  


def main():
  img = cv2.imread("./images/pic1.png", 1)
  out = sequentialChannelEnhancement(img)
  
  muls = cv2.hconcat([img, out])
  utl.cv_show(muls, "Original Image vs Power Gamma Image")   
  return

if __name__ == '__main__':
  main()