import numpy as np
import cv2
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
    
    K = np.minimum(np.minimum(C, M), Y)
    
    denominator = 1 - K
    denominator[denominator == 0] = 1  
    
    C = (C - K) / denominator
    M = (M - K) / denominator
    Y = (Y - K) / denominator
    
    CMYK_image = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)   
    
    return CMYK_image

def main():
    out = []
  
    out.append(cv2.imread("./images/cells.tif")  )
    
    CMYK = bgr_to_cmyk(out[-1])
    out.append(CMYK[:,:,0])
    out.append(cv2.medianBlur(out[-1], 17))
    
    utl.plt_ShowMulsImage(out, "./outs/assignment8/muls_result.png", ["Original","Cyan", "Final Result"])
    utl.plt_show(out[-1], "Result")
    cv2.imwrite("./outs/assignment8/result.png", out[-1])

if __name__ == '__main__': 
    main()