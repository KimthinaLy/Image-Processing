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

def showHistogram(img, fName):
  m,n,space = img.shape
  plt.figure(figsize=(4*space, 2))
  for i in range(space):
    plt.subplot(1, space, i + 1)
    plt.hist(img[:,:,i].ravel(), bins=256,range=(0,255))
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
  plt.savefig(fName)
  plt.show()
  


def main():
  img = cv2.imread("./images/pic1.png", 1)
  showHistogram(img, "./outs/assignment1/histogram.png")
  
  out = sequentialChannelEnhancement(img)
  
  utl.showMultipleImg([img, out], "./outs/assignment1/mulImages.png", "Original Image vs Power Gamma Image")
  
  return

if __name__ == '__main__':
  main()