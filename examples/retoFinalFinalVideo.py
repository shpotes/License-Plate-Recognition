# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 01:24:18 2018

@author: jmunozb
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 22:41:49 2018

@author: jmunozbqq
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 19:38:28 2018

@author: User
"""

import numpy as np
import cv2
import os
import time
from characterSegfun import isPlate

os.getcwd()
os.chdir(r'C:\Users\jmunozb\OneDrive - Universidad EAFIT\Reto AI disruptive')

#Videos
os.chdir(r'C:\Users\jmunozb\OneDrive - Universidad EAFIT\Reto AI disruptive\videos2')
#Imagenes
os.chdir(r'C:\Users\jmunozb\OneDrive - Universidad EAFIT\Reto AI disruptive\2. DATASET PLACAS\1. Placas Originales')

##Control
#control=0 #0 es video, 1 es imagenes

archivos=os.listdir()
os.listdir()
#Trackbar to see Yellow for contour
def nothing(x):
    pass
cv2.namedWindow('image')
cv2.resizeWindow('image',650,600)
cv2.createTrackbar('Hmin','image',0,180,nothing)
cv2.createTrackbar('Smin','image',0,255,nothing)
cv2.createTrackbar('Vmin','image',0,255,nothing)
cv2.createTrackbar('Hmax','image',0,180,nothing)
cv2.createTrackbar('Smax','image',0,255,nothing)
cv2.createTrackbar('Vmax','image',0,255,nothing)
cv2.createTrackbar('Control','image',0,1,nothing)
cv2.createTrackbar('Control2','image',0,1,nothing)
cv2.setTrackbarPos('Hmin','image',10)
cv2.setTrackbarPos('Smin','image',87)
cv2.setTrackbarPos('Vmin','image',93)
cv2.setTrackbarPos('Hmax','image',39)
cv2.setTrackbarPos('Smax','image',255)
cv2.setTrackbarPos('Vmax','image',255)
cv2.createTrackbar('n1','image',0,10000,nothing)
cv2.createTrackbar('peri','image',0,100,nothing)
cv2.createTrackbar('approx','image',4,20,nothing)
cv2.createTrackbar('n2','image',0,100,nothing)
cv2.setTrackbarPos('n1','image',6000)
cv2.setTrackbarPos('n2','image',40)
cv2.setTrackbarPos('approx','image',9)
cv2.setTrackbarPos('peri','image',18)
ret=False
i=0 #contador de archivos
v=0
showtime=3 #tiempo para mostrar imagenes
t_inicial=time.time()
control_previo=0
pausa=False
saveImages=False
recortadas=True
contImages=0

while True:
    #print(cv2.getTrackbarPos('Control','image'))
    saveImages=cv2.getTrackbarPos('Control2','image')
    control=cv2.getTrackbarPos('Control','image')
    n1=cv2.getTrackbarPos('n1','image')
    n2=cv2.getTrackbarPos('n2','image')
    perim=cv2.getTrackbarPos('peri','image')
    approxm=cv2.getTrackbarPos('approx','image')
    if control==control_previo:
        pass
    else:
        i=0
    
    if control==0: #Videos
        os.chdir(r'C:\Users\jmunozb\OneDrive - Universidad EAFIT\Reto AI disruptive\videos2')
        archivos=os.listdir()
        if v==0:
            cap = cv2.VideoCapture(archivos[i])
            ret, frame = cap.read()
            v=1
        else:
            if pausa==False:
                ret, frame = cap.read()
    else:            
        if recortadas:
            os.chdir(r'C:\Users\jmunozb\OneDrive - Universidad EAFIT\Reto AI disruptive\ImagesDetection')                         #Imagenes
        else:
            os.chdir(r'C:\Users\jmunozb\OneDrive - Universidad EAFIT\Reto AI disruptive\2. DATASET PLACAS\1. Placas Originales')
        archivos=os.listdir()
        frame=cv2.imread(archivos[i])
        ret=True
        v=0
    
    control_previo=control
    
    if ret:
        frame=cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA)
        frame_2=frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_or=frame.copy()
        low_green=np.array([cv2.getTrackbarPos('Hmin','image'),cv2.getTrackbarPos('Smin','image'),cv2.getTrackbarPos('Vmin','image')])
        upper_green=np.array([cv2.getTrackbarPos('Hmax','image'),cv2.getTrackbarPos('Smax','image'),cv2.getTrackbarPos('Vmax','image')])
        mask2=cv2.inRange(hsv,low_green,upper_green)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        lower_range = np.array([15,100,100])
        upper_range = np.array([45,255,255])
        
        mask = cv2.inRange(hsv, lower_range, upper_range)

        output = cv2.bitwise_and(frame, frame, mask = mask2)
        im2, contours, hierarchy = cv2.findContours(mask2.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours,key=cv2.contourArea, reverse=True)[:20]
        #  print('contours',len(contours))
        posible_plates=list()
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.023 * peri, True)
            lx,ly,_=frame.shape
            area_frame=lx*ly
            area=cv2.contourArea(cnt)
            """
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            lim_sup=3.9
            lim_inf=1.25
            width,height=np.int0(rect[1])
            area=width*height
            """
            #print(len(approx),peri,area/peri,area/area_frame)
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            # Si el contorno es mas o menos un cuadrado 
            if  3 <len(approx)< approxm and peri > perim and  (1/n1)<(area/area_frame) and (area/area_frame)<(n2/100) :
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                lim_sup=3.5
                lim_inf=1.25
                width,height=np.int0(rect[1])
                if width<height:
                    aux=width
                    width=height
                    height=aux
                dst = np.array([
                        [0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        	], dtype = "float32")
                if rect[2]<-30:
                    dst = np.array([
                        [width - 1, height - 1],
                        [0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        	], dtype = "float32")
                if 135>rect[2]>45 or -135<rect[2]<-45:
                    ## si el ancho es mas de 1.25 el largo pero menos de 3.5
                    if rect[1][1] >lim_inf* rect[1][0] and rect[1][1] <lim_sup* rect[1][0]:
                        cv2.drawContours(frame_or,[box],0,(0,255,0),2)
                        M = cv2.getPerspectiveTransform(np.array(box,dtype="float32"), dst)           
                      #  warp = cv2.warpPerspective(frame, M, (width, height))
                        warp = cv2.warpPerspective(frame_2, M, (width, height))
                        
                        if saveImages:
                            cv2.imwrite('C:/Users/jmunozb/OneDrive - Universidad EAFIT/Reto AI disruptive/ImagesDetectionc/' +str(contImages)+'.png',warp)
                            contImages+=1
                        posible_plates.append(warp)
                  #      warp=cv2.resize(warp,(100,50),interpolation=cv2.INTER_AREA)
                        """
                        if len(approx)==4:
                            
                            M2 = cv2.getPerspectiveTransform(np.array(approx,dtype="float32"), dst)           
                            warp2 = cv2.warpPerspective(frame.copy(), M2, (width, height))
                            print('approx=4',approx,'box',box)
                        else:
                            print('approx',approx)
                        #print(len(approx),peri,area,area/peri)
                        """
                    else:
                        cv2.drawContours(frame_or,[box],0,(255,0,0),2)
                else:
                    if rect[1][0] >lim_inf* rect[1][1] and rect[1][0] <lim_sup* rect[1][1]:
                        cv2.drawContours(frame_or,[box],0,(0,255,0),2)
                        M = cv2.getPerspectiveTransform(np.array(box,dtype="float32"), dst)           
                       # warp = cv2.warpPerspective(frame, M, (width, height))
                        warp = cv2.warpPerspective(frame_2, M, (width, height))
                        
                        if saveImages:
                            cv2.imwrite('C:/Users/jmunozb/OneDrive - Universidad EAFIT/Reto AI disruptive/ImagesDetectionc/' +str(contImages)+'.png',warp)
                            contImages+=1
                        posible_plates.append(warp)
                 #       warp=cv2.resize(warp,(100,50),interpolation=cv2.INTER_AREA)
                        # print(len(approx),peri,area,area/peri)
                        """
                        if len(approx)==4:
                            print('approx=4',approx,'box',box)
                            M2 = cv2.getPerspectiveTransform(np.array(approx,dtype="float32"), dst)           
                            warp2 = cv2.warpPerspective(frame.copy(), M2, (width, height))
                        else:
                            print('approx',approx)
                            """
                    else:
                        cv2.drawContours(frame_or,[box],0,(255,0,0),2)
            else:
                rect = cv2.minAreaRect(cnt)
                #print('bad',len(approx),peri,area/area_frame)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame_or,[box],0,(200,20,200),2)
        
        
        
        
        
        
  
       
       
       
        #try:
         #   warp=cv2.resize(warp,(100,50),interpolation=cv2.INTER_AREA)
          #  cv2.imshow('warped',warp)
           # if saveImages:
            #    cv2.imwrite('C:/Users/jmunozb/OneDrive - Universidad EAFIT/Reto AI disruptive/ImagesDetectionc/' +str(contImages)+'.png',warp)
             #   contImages+=1
                """
            try:
                warp2=cv2.resize(warp2,(100,50),interpolation=cv2.INTER_AREA)
                cv2.imshow('warped2',warp2)
            except:
                print('warp2 not')
                pass
                """
                
        
      #  cv2.imshow('mask2', mask2)
       # cv2.imshow('output',output)
     #   if isPlate(posible_plates):
      #      print('exito')
      #  else:
            #print('fracaso')
        cv2.imshow('frame',frame_or)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(low_green,upper_green)
            break;
        elif cv2.waitKey(1) & 0xFF == ord('p'):
            pausa=not(pausa)
            
        if control_previo==1:
            if (time.time()-t_inicial)>showtime:
            #if (time.time()-t_inicial)>showtime or saveImages:
                t_inicial=time.time()
                i=i+1
                if i>len(archivos)-1:
                    i=0
            else:
                pass
    else:
        if control_previo==1:
            if (time.time()-t_inicial)>showtime:
                i=i+1
                if i>len(archivos)-1:
                    i=0
            else:
                pass
        elif control_previo==0:
            if i<len(archivos)-1:
                v=0
                i=i+1
            else:
                i=0   
                v=0
cap.release()
cv2.destroyAllWindows()
