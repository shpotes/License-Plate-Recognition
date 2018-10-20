# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:28:46 2018

@author: jmunozb
"""

import cv2
import os
import time

archivos = os.listdir(r'C:\Users\jmunozb\OneDrive - Universidad EAFIT\Reto AI disruptive\ImagesDetection')
print('lenarchivos',len(archivos))
i = 0
v = 0
showtime = 3 #tiempo para mostrar imagenes
while True:
    print('archivo','C:/Users/jmunozb/OneDrive - Universidad EAFIT/Reto AI disruptive/ImagesDetection/'+archivos[i])
    img = cv2.imread('C:/Users/jmunozb/OneDrive - Universidad EAFIT/Reto AI disruptive/ImagesDetection/'+archivos[i])
    mser = cv2.MSER_create()
    #Resize the image so that MSER can work better
    nh = img.shape[1]
    nw = img.shape[0]
    img = cv2.resize(img, (nh, nw))
    print(nh*nw)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()

    regions = mser.detectRegions(gray)
    ##resulting bounding boxes 
    hulls = []
    hulls2=[]
    hulls3=[]
    areas = []
    contbbox=0
    for p in regions[0]:
        ax = cv2.contourArea(p)
        a = cv2.convexHull(p.reshape(-1, 1, 2))
        x, y, w2, h2 =regions[1][contbbox]
        contbbox+=1
        ax= w2*h2
        if ax > 0.005*(nh*nw) and ax < 0.08*(nh*nw) and h2>w2 and h2<3*w2:
            print('ax_acepted',ax)
            
            M = cv2.moments(a)
               # calculate x,y coordinate of center
            if    M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                #cX = int(M["m10"] / 1)
                cY = nw
                        
            if   (0.2*nw)<cY<(0.7*nw):
                hulls.append(a)
                cv2.circle(vis, (cX, cY), 1, (0, 255, 0), -1)
            else:
                cv2.circle(vis, (cX, cY), 1, (255, 0, 0), -1)
                hulls2.append(a)
                print('cynh',cX,cY,nw,nh)
        else:
            print('rechazadas',ax)
            hulls3.append(a)
            """
            M = cv2.moments(p)
   # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            """
        cv2.polylines(vis, hulls2, 1, (255,0,0)) 
        cv2.polylines(vis, hulls3, 1, (0,0,255))
        cv2.polylines(vis, hulls, 1, (0,255,0)) 


    cv2.namedWindow('img', 0)
    cv2.imshow('img', vis)
    while(cv2.waitKey()!=ord('q')):
        continue
    i = i + 1
    print(i)
    if i > len(archivos) - 1:
        break
    """
    except:
        i = i + 1
        if i > len(archivos) - 1:
            break
        continue    
        """
print(i)

cv2.destroyAllWindows()