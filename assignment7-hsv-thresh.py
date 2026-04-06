import cv2
import numpy as np
import matplotlib.pyplot as plt
import util as utl
import math

def split4(img):
    h, w, s = img.shape
    h_half = h // 2
    w_half = w // 2

    img_11 = img[:h_half, :w_half]
    img_12 = img[:h_half, w_half:]
    img_21 = img[h_half:, :w_half]
    img_22 = img[h_half:, w_half:]

    return img_11, img_12, img_21, img_22

def Merge4(x, out):
    h, w= out.shape
    h_half = h // 2
    w_half = w // 2
    
    out[:h_half, :w_half] = x[0]
    out[:h_half, w_half:] = x[1]
    out[h_half:, :w_half] = x[2]
    out[h_half:, w_half:] = x[3]
    return out

# ===============Intermean Threshold ==================
def toBinary(gray_img, thresh):
    out= np.zeros_like(gray_img, dtype='uint8')
    out[gray_img >= thresh] = 255
    return out

def intermean_process(hist, t, st, en):
    prob = np.zeros_like(hist, dtype='float16')
    tot = np.sum(hist[st:en])
    prob[st:en] = hist[st:en]/tot
    w0 = np.sum(prob[st:t+1]) + 0.0000001
    w1 = (1 - w0) + 0.0000001
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
    return utl.float2int(thr)
  
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

def main():
    option = [0,1,1,2]
    noise = [1,0,1,0]
    intermean_times = [1,1,1,3]
    out=[]
    
    img = cv2.imread("images/documents.png", 1)
    hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_split = split4(hsv)

    for i in range (len(img_split)):
        local = img_split[i]
        
        if noise[i]:
            local = cv2.medianBlur(local, 3)
            
        x = local[:,:,option[i]]
        if (option[i] == 0 or option[i] == 1):
            x = 255 - x
            
        out.append(toBinary(x, intermean_adaptive(utl.calHist(x), intermean_times[i])))
        
    img_merge = Merge4(out, np.zeros_like(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), dtype='uint8'))
    utl.cv_show(img_merge)
    cv2.imwrite("./outs/assignment7/result.png", img_merge)
    
    return 


if __name__ == '__main__': main()
