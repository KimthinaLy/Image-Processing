import cv2
import numpy as np
import matplotlib.pyplot as plt
import util as utl

def findMask(hue, lower, upper): 
  mask = (hue >= lower) & (hue <= upper) 
  return mask

def applyV_ToMask(hsv, mask, weight):
  hsv[:,:,1][mask] *= weight
  return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def main():
  img = cv2.imread("./images/shade.png", 1)
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
  mask = findMask(hsv[:,:,0]*2, 180, 260)  
  out = applyV_ToMask(hsv, mask, 0.3)
  utl.showMultipleImg([img, out], "./outs/assignment6/result.png", "Original vs Low Saturation")
  
  return

if __name__ == '__main__': main()

#exam blue and yellow

#range1[lower1, upper1], range2[lower2, upper2] 