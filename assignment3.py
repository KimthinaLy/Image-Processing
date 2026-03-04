import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import util as utl

#================== Equalize ========================

def cdfNormalization(hist):
  cdf = hist.cumsum()
  cdf_m = np.ma.masked_equal(cdf, 0)
  cdf_m = ((cdf_m - cdf_m.min())  / (cdf_m.max() - cdf_m.min()))*255
  cdf_m = np.ma.filled(cdf_m, 0).astype('uint8')
  return cdf_m

def equalize(img, hist):  
  cdf_m = cdfNormalization(hist)
  out = cdf_m[img]
  
  return out

#============== Edge Detection =========================

def edge_operator_meth (img, k):
  f = img.copy().astype(np.float32)
  out = np.zeros_like(img, dtype= 'float32')
  mask_gx = np.array([[-1,0,1], [-k,0,k], [-1,0,1]], dtype='float32')
  mask_gy = np.array([[-1,-k,-1], [0,0,0], [1,k,1]], dtype='float32')
  
  sz,sz = mask_gx.shape
  bd = sz // 2
  (m,n) = img.shape
  
  for i in range (bd, m-bd):
    for j in range (bd, n-bd):
      gx, gy = 0., 0.
      sub_f = f[i-bd:bd+i+1, j-bd:bd+j+1]
      gx = np.multiply(sub_f, mask_gx).sum() 
      gy = np.multiply(sub_f, mask_gy).sum()   
      out[i,j] = np.sqrt(gx **2 + gy**2)
  out[out>255.0] = 255.0
  return out.astype(np.uint8)
  

def main():
  out = []
  histList = []
  histTitleList = []
  
  img = cv2.imread("./images/pic3.png", 0)   
  out.append(img)
  histList.append(utl.calHist(out[-1]))
  histTitleList.append("Original Histogram")
  
  out.append(utl.gammaTrans(img,2))
  histList.append(utl.calHist(out[-1]))
  histTitleList.append("Gamma Image Histogram")
  
  out.append(equalize(out[-1], histList[-1]))
  histList.append(utl.calHist(out[-1]))
  histTitleList.append("Equalize Histogram")
  
  out.append(edge_operator_meth(out[-1], 1))
  
  utl.plotMultipleHist(histList, histTitleList, "./outs/assignment3/mulHistograms.png")
  
  utl.showMultipleImg(out,"./outs/assignment3/mulImages.png" ,"Original vs Power Gamma vs Equalize Image vs Edge Image")
  utl.cv_show(out[-1], "Edge Image")

  return 

if __name__ == "__main__": main()