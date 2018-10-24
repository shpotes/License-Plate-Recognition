'''
Batman team solution for license plate detection
'''

import numpy as np
import cv2
import os
import time

archivos=os.listdir()
os.listdir()

def isPlate(img, i):
    plate = []
    saveChar = True
    contImages2 = 0
    
    mser = cv2.MSER_create()
    # gray image
    vis = img.copy()
    modified = tres(vis)
    # collect bounding boxes from the image
    regions = mser.detectRegions(modified)
    bboxes = identifyBoundingBoxes(regions,vis)
    clases = calcularClases(bboxes)
    sorted_clases = sorted(clases,key=lambda x: x[0])
    
    if len(clases) == 6:
        for areas2 in sorted_clases:
            if saveChar:
                imgSaved = modified[areas2[1]:areas2[1]+areas2[3],areas2[0]:areas2[0]+areas2[2]]
                #_, inv = cv2.threshold(imgSaved, 127,255,cv2.THRESH_BINARY_INV)
                cv2.imwrite(str(contImages2)+str(i)+'.png',imgSaved)
                plate.append(imgSaved)
                contImages2+=1
                
            else:
                cv2.rectangle(vis,(areas2[0],areas2[1]),(areas2[0]+areas2[2],areas2[1]+areas2[3]),(255,255,255))

    return plate

def identifyBoundingBoxes(regions, vis):
    nh = vis.shape[1]
    nw = vis.shape[0]
    ltmp = list(regions[1])
    sorted_y_idx_list = sorted(range(len(ltmp)),key=lambda x: ltmp[x][0])
    nregions1 = [ltmp[i] for i in sorted_y_idx_list ]
    nregions0 = [regions[0][i] for i in sorted_y_idx_list ]
    
    bboxes = []
    contbbox = 0

    for p in nregions0:
        ax = cv2.contourArea(p)
        a = cv2.convexHull(p.reshape(-1, 1, 2))
        x, y, w2, h2 = nregions1[contbbox]
        ax = w2*h2
        
        if ax > 0.02*(nh*nw) and ax < 0.16*(nh*nw) and h2>w2 and h2<3*w2:
            cv2.rectangle(vis,(x,y),(x+w2,y+h2),(255,20,25))
            # Calculate the center of the region
            M = cv2.moments(a)
            if M["m00"] != 0:
                cY = int(M["m01"] / M["m00"])
            else:
                cY = nw

            if (0.25*nw) < cY < (0.65*nw):
                # append the accepted bounding box
                bboxes.append(nregions1[contbbox])
                
        contbbox+=1 
    
    return bboxes

def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def calcularClases(bboxes):
    clases = [[]]
    if bboxes == []:
        return []
    for b in bboxes:
        for i in range(len(clases)):
            l = clases[i]
            if l == []:
                l.append(b)
            else:
                ll = l[-1]
                db = {'x1':b[0], 'x2':b[0]+b[2], 'y1':b[1], 'y2':b[1]+b[3]}
                dl = {'x1':ll[0], 'x2':ll[0]+ll[2], 'y1':ll[1], 'y2':ll[1]+ll[3]}
                ii = get_iou( db, dl)
                if ii > 0.05:
                    l.append(b)
                    break
                else:
                    if i == len(clases)-1:
                        clases.append([b])
                        break
    clas = [0]*(len(clases))
    for i in range(len(clases)):
        l = clases[i]
        clas[i] = max(l, key = lambda x: x[1]*x[2])
    
    return clas    

def tres(img):
    grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 1)
    return th

# Model parameters
low_green=np.array([10,87,93])
upper_green=np.array([39,255,255])

n1 = 6000
n2 = 40
approxm = 9
perim = 18
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

plates = []
# Video name
vn = 'StreetView.mp4'

while True:
    
    if v == 0:
        cap = cv2.VideoCapture(vn)
        ret, frame = cap.read()
        v = 1
    else:
        ret, frame = cap.read()    
    
    if ret:
        frame=cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA)
        frame_2=frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_or=frame.copy()
        mask2=cv2.inRange(hsv,low_green,upper_green)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        lower_range = np.array([15,100,100])
        upper_range = np.array([45,255,255])
        
        mask = cv2.inRange(hsv, lower_range, upper_range)

        output = cv2.bitwise_and(frame, frame, mask = mask2)
        im2, contours, hierarchy = cv2.findContours(mask2.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours,key=cv2.contourArea, reverse=True)[:20]
        
        posible_plates = list()
        
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.023 * peri, True)
            lx,ly,_ = frame.shape
            area_frame = lx*ly
            area = cv2.contourArea(cnt)
            
            if 3 <len(approx)< approxm and peri > perim and (1/n1)<(area/area_frame) and (area/area_frame)<(n2/100) :
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                lim_sup=3.5
                lim_inf=1.25
                width,height=np.int0(rect[1])
                if width < height:
                    aux = width
                    width = height
                    height = aux
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
                        
                        posible_plates.append(warp)
                  
                    else:
                        cv2.drawContours(frame_or,[box],0,(255,0,0),2)
                else:
                    if rect[1][0] >lim_inf* rect[1][1] and rect[1][0] <lim_sup* rect[1][1]:
                        cv2.drawContours(frame_or,[box],0,(0,255,0),2)
                        M = cv2.getPerspectiveTransform(np.array(box,dtype="float32"), dst)           
                       # warp = cv2.warpPerspective(frame, M, (width, height))
                        warp = cv2.warpPerspective(frame_2, M, (width, height))
                        
                        posible_plates.append(warp)
                 
                    else:
                        cv2.drawContours(frame_or,[box],0,(255,0,0),2)
            else:
                rect = cv2.minAreaRect(cnt)
                #print('bad',len(approx),peri,area/area_frame)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame_or,[box],0,(200,20,200),2)
        
    else:
        break
    for i in range(len(posible_plates)):
        a = isPlate(posible_plates[i], i)
        if len(a) != 0:
            plates.append(a)

cap.release()
cv2.destroyAllWindows()


# the plates aray contains lists with the 6 characters of the plates