import cv2
import numpy as np
import matplotlib.pyplot as plt
import util as utl  
  
def cdfNormalization(hist):
  cdf = hist.cumsum()
  cdf_m = np.ma.masked_equal(cdf, 0)
  cdf_m = ((cdf_m - cdf_m.min())  / (cdf_m.max() - cdf_m.min()))*255
  cdf_m = np.ma.filled(cdf_m, 0).astype('uint8')
  return cdf_m

def mappingWithCdfm( img, cdf_m):
  r,c = img.shape
  out = np.zeros_like(img, dtype='uint8')
  
  for i in range(r):
    for j in range(c):
      intens = img[i,j]
      out[i,j] = cdf_m[intens]
  return out

def equalize(img, hist):
  cdf_m = cdfNormalization(hist)
  #out = cdf_m[img]
  out = mappingWithCdfm(img, cdf_m)
  
  return out

def main():
  img = cv2.imread("./images/pic2.png",0)
  hist = utl.calHist(img)
  
  out = equalize(img, hist)
  
  muls = cv2.hconcat([img, out])
  utl.cv_show(muls, "Original vs Equalized")
  utl.plotMultipleHist([hist, utl.calHist(out)], ["Original Histogram", "Equalized Histogram"])
  
  return

if __name__ == '__main__': main() 