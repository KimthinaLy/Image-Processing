import cv2
import numpy as np
import matplotlib.pyplot as plt 

def toGrayscale(img):
  b = img[:,:,0]
  g = img[:,:,1]
  r = img[:,:,2]
  m,n = r.shape 
  
  gray = np.zeros((m,n), dtype='float16' )
  for i in range (m):
    for j in range (n):
      gray[i,j] = 0.2989*r[i,j] + 0.5870*g[i,j] + 0.1140*b[i,j]
  
  gray = gray.astype(np.uint8)
  return gray

def plt_show(img, title = "Xxxx"):
  if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  plt.imshow(img, cmap = 'gray')
  plt.axis('off')
  plt.title(title)
  plt.show()
  
def cv_show(img, title="Xxxx"):
  cv2.imshow(title, img)
  cv2.waitKey()
  cv2.destroyAllWindows()
  
def gammaTrans(gray_img, gamma):
  c = 255.0
  img_norm = gray_img.astype(np.float16)
  img_norm = img_norm/img_norm.max()
  gamma_img = (img_norm ** gamma) * c
  return gamma_img.astype(np.uint8)

def calHist(img):
  hist = np.zeros(256, dtype=int)
  m,n = img.shape
  for i in range(m):
    for j in range(n):
      intensity = img[i,j]
      hist[intensity] += 1
  return hist

def showHist(hist, title="Histogram"):
  plt.figure(figsize=(4,2))
  plt.plot(hist)
  plt.xlabel('Pixel Intensity')
  plt.ylabel('Frequency')
  plt.title(title)
  plt.show()

  
def plotMultipleHist(histList, titleList, fName):
  n = len(histList)
  plt.figure(figsize=(4*n, 2))
  
  for i in range(n):
    plt.subplot(1, n, i + 1)
    plt.plot(histList[i])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(titleList[i])
  plt.savefig(fName)
  plt.show()
  
def showMultipleImg(imgList,fName, title = "Xxxx"):
  n = len(imgList)
  muls = []
  
  for i in range (n):
    img = imgList[i]
    if len(img.shape) == 3:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    muls.append(img)

  muls = cv2.hconcat(muls)
  plt.imshow(muls, cmap = 'gray')
  plt.axis('off')
  plt.title(title)
  plt.show()
  muls = cv2.cvtColor(muls, cv2.COLOR_RGB2BGR)
  cv2.imwrite(fName, muls.astype(np.uint8))