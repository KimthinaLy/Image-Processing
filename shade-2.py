import cv2
import numpy as np
import matplotlib.pyplot as plt
import util as utl

def findMask(hue, range0, range1): 
  mask0 = (hue >= range0[0]) & (hue <= range0[1]) 
  mask1 = (hue >= range1[0]) & (hue <= range1[1]) 
  return mask0, mask1

def applyV_ToMask(hsv, mask, w1):
  hsv[:,:,0][mask[0]] = 120

  hsv[:,:,1][mask[1]] *= w1  
  return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def main():
  img = cv2.imread("./images/shade.png", 1)
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
  mask = findMask(hsv[:,:,0]*2, (180, 270), (10, 80))  
  out = applyV_ToMask(hsv, mask,  0.3)
  utl.showMultipleImg([img, out], "./outs/exam-prep/shade-result.png", "Original - Result")
    

if __name__ == '__main__': main()