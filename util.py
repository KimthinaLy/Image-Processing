import cv2
import numpy as np
import matplotlib.pyplot as plt 
import math


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
  gamma_img = gray_img.astype(np.float16)
  return (((gamma_img/gamma_img.max())**gamma) * c).astype(np.uint8)

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
  plt.tight_layout()
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
  plt.tight_layout()
  plt.show()
  muls = cv2.cvtColor(muls, cv2.COLOR_RGB2BGR)
  cv2.imwrite(fName, muls.astype(np.uint8))
  
def float2int(x):
    if x - math.floor(x) >= 0.5:
        result = math.ceil(x)
    else:
        result = math.floor(x)
    return result

def plt_ShowMulsImage(imgList,fName,title):
    n = len(imgList)
    for i in range(n):
        if len(imgList[i].shape) == 3:
          imgList[i] = cv2.cvtColor(imgList[i], cv2.COLOR_BGR2RGB)
        plt.subplot(1, n, i + 1) 
        plt.imshow(imgList[i], cmap='gray')
        plt.title(title[i])
        plt.axis('off')
    plt.savefig(fName)
    plt.tight_layout()
    plt.show()
    
#================= intermean call by adaptive_intermean or toBinary===============
def toBinary(gray_img, thresh):
    out= np.zeros_like(gray_img, dtype='uint8')
    out[gray_img >= thresh] = 255
    return out

def intermean_process(hist, t, st, en):
    prob = np.zeros_like(hist, dtype='float16')
    tot = np.sum(hist[st:en])
    prob[st:en] = hist[st:en]/tot
    w0 = np.sum(prob[st:t+1]) + 0.0000001
    w1 = (1 - w0) +  + 0.0000001
    i0 = np.array([i for i in range(st,t+1)])
    i1 = np.array([i for i in range(t+1, en)])
    u0 = np.sum(i0*prob[st:t+1])/w0
    u1 = np.sum(i1*prob[t+1:en])/w1
    if u0 == 0.0:
        thr = u1
    elif u1 == 0.0:
        thr = u0
    else:
        thr = (u0+u1)/2
    return float2int(thr)
  
def intermean_method(hist,st,en):
    T = []
    t0 = int((st+en)*0.5)
    T.append(t0)

    max_iter = 10
    for i in range (max_iter): #prevent infinite loop
        t1 = intermean_process(hist, t0, st, en)
        T.append(t1)
        if abs(t1-t0) <= 2 :
            break
        else:
            t0 = t1
    thr = T[-1]
    print(T, thr)
    return thr 

def intermean_adaptive(hist, times):
    threshs = -1
    st, en = 0, 256
    for inx in range(times):
        #plot_histogram(hist,st,en)
        threshs = intermean_method(hist, st, en)
        en = threshs+1
    return threshs
  
#================ CMYK =====================
def bgr_to_cmyk(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #can also astype to float here
    rgb_normalized = rgb_image / 255.0
    
    R = rgb_normalized[:, :, 0]
    G = rgb_normalized[:, :, 1]
    B = rgb_normalized[:, :, 2]

    # Calculate the CMY values
    C = 1 - R
    M = 1 - G
    Y = 1 - B
    
    K = np.minimum(np.minimum(C, M), Y)
    
    denominator = 1 - K
    denominator[denominator == 0] = 1  
    
    C = (C - K) / denominator
    M = (M - K) / denominator
    Y = (Y - K) / denominator
    
    CMYK_image = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)   
    
    return CMYK_image