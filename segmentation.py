import cv2
import os
import time

archivos = os.listdir('placas')
i = 0
v = 0
showtime = 3 #tiempo para mostrar imagenes
while True:
    try:
        img = cv2.imread(archivos[i])
        mser = cv2.MSER_create()
        #Resize the image so that MSER can work better
        nh = img.shape[1]*2
        nw = img.shape[0]*2
        img = cv2.resize(img, (nh, nw))
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis = img.copy()

        regions = mser.detectRegions(gray)

        hulls = []
        areas = []
        for p in regions[0]:
            a = cv2.convexHull(p.reshape(-1, 1, 2))
            ax = cv2.contourArea(p)
            if ax > 0.03*(nh*nw) and ax < 0.20*(nh*nw):
                hulls.append(a)
            
            cv2.polylines(vis, hulls, 1, (0,255,0)) 


        cv2.namedWindow('img', 0)
        cv2.imshow('img', vis)
        while(cv2.waitKey()!=ord('q')):
            continue
        i = i + 1
        print(i)
        if i > len(archivos) - 1:
            break
    except:
        i = i + 1
        if i > len(archivos) - 1:
            break
        continue    

cv2.destroyAllWindows()
