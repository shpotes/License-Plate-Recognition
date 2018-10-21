# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:28:46 2018

@author: jmunozb
"""

import cv2
import os
import time
from sklearn.cluster import KMeans
import numpy as np

def isPlate(archivos):
    a_Plate=False
#archivos = os.listdir(r'C:\Users\jmunozb\OneDrive - Universidad EAFIT\Reto AI disruptive\ImagesDetection')
#print('lenarchivos',len(archivos))
    i = 0
    v = 0
    saveChar=True
    contImages2=0
    
    kernel = np.ones((2,2),np.uint8)
    showtime = 3 #tiempo para mostrar imagenes
    for img in  archivos:
        #print('archivo','C:/Users/jmunozb/OneDrive - Universidad EAFIT/Reto AI disruptive/ImagesDetection/'+archivos[i])
        #img = cv2.imread('C:/Users/jmunozb/OneDrive - Universidad EAFIT/Reto AI disruptive/ImagesDetection/'+archivos[i])
        mser = cv2.MSER_create()
        #Resize the image so that MSER can work better
        nh = img.shape[1]
        nw = img.shape[0]
        img = cv2.resize(img, (nh, nw))
        #print(nh*nw)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis = img.copy()
    
        regions = mser.detectRegions(gray)
        ltmp=list(regions[1])
        sorted_y_idx_list = sorted(range(len(ltmp)),key=lambda x: ltmp[x][0])
    
        nregions1 = [ltmp[i] for i in sorted_y_idx_list ]
        nregions0= [regions[0][i] for i in sorted_y_idx_list ]
      #  print( "Xs:", Xs )
        #[print(x,y)for x,y in zip(regions[0],regions[1])]
        """
        tmp = []
        for i in range(len(regions[0])):
            tmp.append([regions[0][i], regions[1][i]])
        tmp.sort(key=lambda x: x[1][-2] * x[1][-1])
        """
        #list1, list2 = (list(t) for t in zip(*sorted(zip(regions[0], regions[1]))))
        
        ##resulting bounding boxes 
        hulls = []
        hulls2=[]
        hulls3=[]
        areas = []
        contbbox=0
        epsilon=0.06*nh
        #print('nh',nh)
        coordenaditas=[x[0] for x in nregions1]
        c2=np.array(coordenaditas).reshape(-1,1)
        #kmeans=KMeans(n_clusters=6,random_state=0).fit(c2)
        #my_labels=kmeans.labels_
        goodBox=list()
        current_l=-10
        cont_km=0
        firstkm=0
        my_areas=list()
        my_areas2=list()
        cont_a2=0
        first_m2=1
        
        #regions[1].sort(key=lambda x : x[2]*x[3] )
        #print(nregions1)
        for p in nregions0:
            ax = cv2.contourArea(p)
            a = cv2.convexHull(p.reshape(-1, 1, 2))
            x, y, w2, h2 =nregions1[contbbox]
            ax= w2*h2
            """
            if my_labels[contbbox] == current_l :
                if ax  > my_areas[cont_km] :
                    my_areas[cont_km]=ax
            else:
                if firstkm==0:
                    firstkm=1
                    current_l=my_labels[contbbox]
                    my_areas.append(ax)
                else:   
                    current_l=my_labels[contbbox]
                    my_areas.append(ax)
                    
                    cont_km+=1
                    """
            
            
            
            if ax > 0.02*(nh*nw) and ax < 0.16*(nh*nw) and h2>w2 and h2<3*w2:
                cv2.rectangle(vis,(x,y),(x+w2,y+h2),(255,20,25))
              #  print('ax_acepted',ax)
                
                M = cv2.moments(a)
                   # calculate x,y coordinate of center
                if    M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    #cX = int(M["m10"] / 1)
                    cY = nw
                            
                if   (0.25*nw)<cY<(0.65*nw):
                    if first_m2:
                        #print('firstm2')
                        first_m2=0
                        my_areas2.append(nregions1[contbbox])
                    else :
                        #print('firstm3',x,epsilon)
                        if (x-my_areas2[cont_a2][0]) < epsilon:
                            if ax > (my_areas2[cont_a2][2]*my_areas2[cont_a2][3]):
                                my_areas2[cont_a2]=nregions1[contbbox]
                        else:
                                my_areas2.append(nregions1[contbbox])
                                cont_a2 +=1
                    hulls.append(a)
                    #cv2.circle(vis, (cX, cY), 1, (0, 255, 0), -1)
                    #print('centroides',cX,cY)
                   # print('my_areas2',my_areas2)
                #else:
                 #   continue:
                    #cv2.circle(vis, (cX, cY), 1, (255, 0, 0), -1)
                  #  hulls2.append(a)
                #    print('cynh',cX,cY,nw,nh)
            #else:
                
               # print('rechazadas',ax)
              #  hulls3.append(a)
                """
                M = cv2.moments(p)
       # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"]) """
            contbbox+=1   
            """
            cv2.polylines(vis, hulls2, 1, (255,0,0)) 
            cv2.polylines(vis, hulls3, 1, (0,0,255))
            cv2.polylines(vis, hulls, 1, (0,255,0)) 
            
            """
        if len(my_areas2)==6:
            a_Plate=True
            print('entró')
            for areas2 in my_areas2:
                if saveChar:
                    lower_range = np.array([0,0,0])
                    upper_range = np.array([180,255,80])
                    lower_range2 = np.array([0,0,0])
                    upper_range2 = np.array([70,70,70])
                    #cvInRangeS(imgHSV, cvScalar(0, 0, 0, 0), cvScalar(180, 255, 30, 0), imgThreshold);
                    imgSaved=vis[areas2[1]:areas2[1]+areas2[3],areas2[0]:areas2[0]+areas2[2]]
                    hsv = cv2.cvtColor(imgSaved, cv2.COLOR_BGR2HSV)
          #          try:
                    maskf = cv2.inRange(src=hsv, lowerb=lower_range, upperb=upper_range)
                    cv2.rectangle(vis,(areas2[0],areas2[1]),(areas2[0]+areas2[2],areas2[1]+areas2[3]),(0,255,0))
                
                    #maskf = cv2.morphologyEx(maskf, cv2.MORPH_CLOSE, kernel)
                    _,thresh2 = cv2.threshold(maskf,127,255,cv2.THRESH_BINARY_INV)
                    cv2.imwrite('C:/Users/jmunozb/OneDrive - Universidad EAFIT/Reto AI disruptive/char/' +str(contImages2)+'.png',thresh2)
                    contImages2+=1
                    cv2.imshow('img2', maskf)
                    cv2.waitKey(0)
                    """while(cv2.waitKey(1)!=ord('q')):
                    continue"""
                  #  cv2.waitKey(1000) & 0xFF == ord('p')
                        
                    """
                    except:
                        print('nodio',contImages2)
                        print(lower_range,upper_range)
                        pass
                        """
                        
                else:
                    cv2.rectangle(vis,(areas2[0],areas2[1]),(areas2[0]+areas2[2],areas2[1]+areas2[3]),(255,255,255))
       # else:
           # print('lenareas',len(my_areas2))
       # print(kmeans.labels_)
        if len(my_areas2)==6:
            print('buena')
            """
            cv2.imshow('imgfull', vis)
            """
            #while(cv2.waitKey(1)!=ord('q')):
               # continue"""
            #cv2.waitKey(1000) & 0xFF == ord('p')
            """

    
        except:
            i = i + 1
            if i > len(archivos) - 1:
                break
            continue    
            """
    #print(i)
    #if len (archivos) > 1:
        #cv2.destroyAllWindows()
    return a_Plate