"""
Created on Tue Jul 27 10:18:46 2021

@author: hm
"""

from glob import glob
import time
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from torchvision import transforms
from PIL import Image
from threading import Thread
import psutil
import cv2
import sys
import numpy as np
import statistics as st
from copy import deepcopy
import pandas as pd
import traceback
from utils import faceMTCNN, embedResnet, DeepPixBiSExtractor
plt.ioff()


demo_length = 100 #frame
detector = faceMTCNN(period=1)
verifier = embedResnet(period=3)

liveness = DeepPixBiSExtractor(scoring_method='combined', #['pixel_mean','binary','combined']
                            model_file='Pretrained_models/OULU_Protocol_1_model_0_0.pth')

tensor2pil = transforms.ToPILImage(mode='RGB')
cpu_hist = []
cpu_ma1  = []
cpu_ma2  = []
stamp    = []
stop_threads = False
def cpu_usage(delay=.1,ma_win1=10,ma_win2=100):
    t = 0
    while True:
        cusage = psutil.cpu_percent()
        cpu_hist.append(cusage)
        cpu_ma1.append(st.mean(cpu_hist[-ma_win1:]))
        cpu_ma2.append(st.mean(cpu_hist[-ma_win2:]))
        stamp.append(t)
        t += delay
        time.sleep(delay)
        global stop_threads 
        if stop_threads: 
            print('kill hist thread')
            break
        
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (15,35)
fontScale              = 1
lineType               = 2        

wait = .2
th1 = .85
th2 = 1.1
i = 0

ss = 100
# time.sleep(5)
thread = Thread(target = cpu_usage,)
thread.start()
# time.sleep(15)
w,h = (1280//2,720//2)
fps = 1
buffer = np.ones((w,w,3))*255
vid = cv2.VideoCapture("""autovideosrc 
                       ! videoconvert 
                       ! video/x-raw, framerate=5/1,
                       width=640, height=480, format=BGR 
                       ! appsink""")
# vid  = cv2.VideoCapture(0)
while True:
    ret,first_frame = vid.read()
    time.sleep(wait)
    if ret:
        break
print('get frist frame!')
first_frame = cv2.resize(first_frame,(w,h))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
h,w = tuple(first_frame.shape[:2])
# cv2.namedWindow('res',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('res', w,w+h)
video = cv2.VideoWriter('demo.avi',fourcc,5,(w,w+h))
facebank = pd.read_csv('hm.csv',header=None)
verify = False
norm_min  = 1.0
norm_mean = 1.0

print('begin real-time process')
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, w)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
# vid.set(cv2.CAP_PROP_FPS,fps)
tracker = cv2.TrackerKCF_create()
init_tracker = False
tight = None
liveness_score = 0
live_turn = 0
sequence = []
while True:
    time.sleep(wait)
    ret,frame = vid.read()
    frame = cv2.resize(frame,(w,h))[:,:,::-1]
    output = frame.copy()
    res    = frame.copy()
    orig   = frame.copy()
    verify_status = 'NOT VERIFIED'
    if ret:
        # print(f'{i}-frame, time:{stamp[-1]}s')
        img = Image.fromarray(frame)
        i += 1
        if i % 10 == 0 :
            print('i:\t',i)
        if i >= demo_length :
            print('end of demo!')
            break
        # detection
        try:
            img_cropped, boxes = detector.schedule(img)
        except Exception as e:
            traceback.print_exc()
            print("error:",e)
            print('\n')
            sequence.append(frame)
            continue
        if img_cropped is not None:
            verifier.last_face = img_cropped
        # verification
        embeddings = verifier.schedule()
        if embeddings is not None:
            last_verify = None
            diff = facebank - embeddings
            norm = np.linalg.norm(diff,axis=1)
            norm_min = np.around(norm.min(),3)
            norm_mean = np.around(norm.mean(),3)
            
            live_turn += 1 
            # liveness
            if tight is not None and live_turn >= 2:
                time.sleep(1.5)
                liveness_score  =  liveness(tight)
                live_turn = 0 
            
            
            if norm.min() < th1 and norm.mean() < th2 :
                verify = True
            else:
                verify = False
        else:
                last_verify = verify
                if not last_verify:
                    try:
                        img_cropped, boxes = detector.detect(img)
                    except Exception as e:
                        traceback.print_exc()
                        print("error:",e)
                        print('\n')
                        sequence.append(frame)
                        continue
                    try:
                        print(type(img_cropped))
                        embeddings = verifier.verify(img_cropped)
                        last_verify = None
                        diff = facebank - embeddings
                        norm = np.linalg.norm(diff,axis=1)
                        norm_min = np.around(norm.min(),3)
                        norm_mean = np.around(norm.mean(),3)
                    except:
                        continue
                    if norm.min() < th1 and norm.mean() < th2 :
                        verify = True
                    else:
                        verify = False
                    
                    
        
        #visualize webcam and verification
        if last_verify is not None:
            if verify == True:
                verify_status = f'VERIFIED min:{norm_min}, mean:{norm_mean}'
                color = (255,255,0)
            elif verify == False:
                verify_status = f'NOT VERIFIED min:{norm_min}, mean:{norm_mean}'
                color = (255,0,0)
                
        else:   
            if verify == True:
                verify_status = f'VERIFIED min:{norm_min}, mean:{norm_mean}'
                color = (0,255,0)
                
            elif verify == False:
                verify_status = f'NOT VERIFIED min:{norm_min}, mean:{norm_mean}'
                color = (255,0,0)
                
        if boxes is not None:
            x1,y1,x2,y2 = np.array(boxes[0]).astype(int)
            tight_crop = orig[y1:y2,x1:x2]
            tight = cv2.resize(tight_crop,(224,224)).transpose(2,0,1)
            res = cv2.rectangle(res,(x1,y1),(x2,y2),color,2)
            w_face, h_face = (x2 - x1),(y2 - y1)
            if init_tracker == False:
                ok = tracker.init(frame, tuple((x1,y1,w_face,h_face)))
                init_tracker = ok
            else:
                tracker = cv2.TrackerKCF_create()
                ok = tracker.init(frame, tuple((x1,y1,w_face,h_face)))
                
        elif init_tracker:
            ok, bbox = tracker.update(frame)
            x1,y1,w_face, h_face= np.array(bbox).astype(int)
            res = cv2.rectangle(res,(x1,y1),(x1+w_face,y1+h_face),color,2)
            
        res = cv2.putText(  res,verify_status, 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            color,
                            lineType 
                            )

        res = cv2.putText(  res,f'liveness: {liveness_score:.3f}', 
                            (15,35+35), 
                            font, 
                            fontScale,
                            color,
                            lineType 
                            )
        
        sequence.append(res)
        
        
          
    else:
        break
    
plt.close()
#cv2.destroyAllWindows()

stop_threads = True
print('last stamp:',stamp[-1])
print(len(stamp),len(sequence))
fig = plt.figure()
ss = 0
for r,res in enumerate(sequence):
    # visualize cpu usage
    # r *= 1
    if ss < 100:
        ss += 1
    plt.plot(stamp[r:r+ss],cpu_hist[r:r+ss],linewidth=1,alpha=0.2)
    plt.plot(stamp[r:r+ss],cpu_ma1[r:r+ss],linewidth=1,c='black',alpha=0.7,linestyle='--')
    plt.plot(stamp[r:r+ss],cpu_ma2[r:r+ss],linewidth=2,c='#38a7ab',
             path_effects=[pe.Stroke(linewidth=2, foreground='g'), pe.Normal()])
    plt.title('CPU Usage')
    plt.savefig('buffer.jpg')
    buffer = cv2.imread('buffer.jpg')
    buffer = cv2.resize(buffer,(w,w))
    output = np.vstack((res[:,:,::-1],buffer))
    video.write(output)
    fig.clear()
vid.release()


