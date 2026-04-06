import cv2
import numpy as np
import matplotlib.pyplot as plt
import util as utl

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
    
    K = np.minimum(C,M,Y)
    
    denominator = 1 - K
    denominator[denominator == 0] = 1  
    
    C = (C - K) / denominator
    M = (M - K) / denominator
    Y = (Y - K) / denominator
    
    CMYK_image = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)   
    
    return CMYK_image


def main():
    img = cv2.imread("./images/bank4.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    
    s = 255-s
    hist = utl.calHist(s)
    thr = utl.intermean_adaptive(hist, 2)
    out = utl.toBinary(s, thr)
    utl.cv_show(out)

    
    return

if __name__ == '__main__': main()
