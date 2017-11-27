import cv2
import numpy as np
import imutils
import math
import random

from matplotlib.pyplot import gray

def drawLine(line,img,rgb=(0,0,255)):
    if(line[1]!=0):

        m=-1/np.tan(line[1])
        c=line[0]/np.sin(line[1])
        cv2.line(img,(0,int(c)),(int(img.shape[0]),int(m*img.shape[0]+c)),rgb)
    else:
        cv2.line(img,(line[0],0),(line[0],img.shape[1]),rgb)


def nothing(x):
    pass

images=["sudoku-original.jpg"]
# Load an image

images=[cv2.imread(img) for img in images]
# Resize The image
images=[imutils.resize(img, width=600) for img in images]




while(1):
    stack=[]
    all=[]
    processed=[]
    i=0
    for img in images:

            contour_list=[];
            max=1000

            clone=0
            pic=0


            clone = img.copy()

            #cv2.addWeighted(clone,i/10,np.zeros(img.shape,img.dtype),0,100)
            gray = cv2.cvtColor(cv2.addWeighted(clone,2,np.zeros(img.shape,img.dtype),0,50), cv2.COLOR_BGR2GRAY)

            gray = cv2.GaussianBlur(gray,(11,11),0)
            gray_threshed2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,2)
            kernel = [[0,1,0],[1,1,1],[0,1,0]]
            kernel=np.array(kernel)

            gray_threshed2=cv2.dilate(gray_threshed2,kernel,iterations=1)
            _, contours, _= cv2.findContours(gray_threshed2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
            count=0
            max_blob=-1
            max_blob_size=-1
            max_ptr=(0,0)
            for y in range(0,len(gray_threshed2)):
                row=gray_threshed2[y]
                for x in range(0,len(row)):
                    if(row[x]>=128):
                        h, w = gray_threshed2.shape[:2]
                        mask = np.zeros((h+2, w+2), np.uint8)
                        area = cv2.floodFill(gray_threshed2,mask,(x,y),64)

                        if(area[0]>max_blob):
                            max_ptr=(x,y)
                            max_blob=area[0]
            h, w = gray_threshed2.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(gray_threshed2,mask,max_ptr,256)
            for y in range(0,len(gray_threshed2)):
                row=gray_threshed2[y]
                for x in range(0,len(row)):
                    if (row[x]==64 and x!=max_ptr[0] and y!=max_ptr[1]):
                        h, w = gray_threshed2.shape[:2]
                        mask = np.zeros((h+2, w+2), np.uint8)
                        area = cv2.floodFill(gray_threshed2,mask,(x,y),0)
            gray_threshed2=cv2.erode(gray_threshed2,kernel,iterations=1)

            lines=cv2.HoughLines(gray_threshed2,1, np.pi/180, 200);
            for i in range(0,len(lines)):

                drawLine(lines[i][0],gray_threshed2,(128,0,128))
            pic=gray_threshed2

    # Draw contours on the original image

    # there is an outer boundary and inner boundary for each eadge, so contours double

            processed.append(pic)

    stack_1=np.vstack((processed))


    cv2.imshow('Objects Detected_1',processed[0])

    cv2.waitKey()

cv2.destroyAllWindows()
