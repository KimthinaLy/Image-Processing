import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import util as utl

def horizonalFlip(img):
  out = np.zeros_like(img, dtype='uint8')
  r,c = img.shape
  for i in range(r):
    for j in range(c):
      out[i,j] = img[i, c-j-1]
  return out

def centroid_image(img):
    r,c = img.shape #r,c,n = img.shape
    return (int(r/2),int(c/2)) 

def init_Transformation():
    return np.identity(3, dtype = float)

def matrix_Translate(T, tx, ty):
    Ts =  np.identity(3, dtype = float)
    Ts[2,0] = tx
    Ts[2,1] = ty
    return np.matmul(Ts, T)

def matrix_Scale(T, sx, sy):
    S =  np.identity(3, dtype = float)
    S[0,0] = sx
    S[1,1] = sy
    return  np.matmul(S, T)    
  
def matrix_Rotatef(T, theta):
    R =  np.identity(3, dtype = float)
    ang = (theta*np.pi)/180
    R[0,0] = math.cos(ang)
    R[0,1] = math.sin(ang)
    R[1,0] = -math.sin(ang)
    R[1,1] = math.cos(ang)
    return np.matmul(R, T)
  
def img_transform(img, T):
    out = np.zeros_like(img, dtype='uint8')
    rows, cols = img.shape
    for y in range(rows):
        for x in range(cols):
            xy = np.array([x, y, 1], dtype = float)
            new_xy = np.matmul(xy, T)
            xn = int(new_xy[0])
            yn = int(new_xy[1])
            if 0 <= xn < cols and 0 <= yn < rows:
                out[yn,xn] = img[y,x]
    return out.astype(np.uint8)
  
def main():
  out = []
  img = cv2.imread("./images/cameraman.png",0)
  out.append(img)
  
  out.append(horizonalFlip(img))

  cen=centroid_image(img)
  T = init_Transformation()
  T= matrix_Translate(T, cen[0], cen[1])
  T = matrix_Scale(T, 0.5, 0.5)
  T = matrix_Rotatef(T, -45)
  T= matrix_Translate(T, -cen[0], -cen[1])
  out.append(img_transform(out[-1], T))
  
  
  utl.cv_show(out[-1], "Result") 
  utl.showMultipleImg(out, "./outs/assignment5/mul-images.png", "Original vs Flipped vs Transformed")
  
  return 

if __name__ == '__main__': main()