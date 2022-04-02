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
from utils import faceMTCNN, embedResnet
plt.ioff()

# If required, create a face detection pipeline using MTCNN:
detector = faceMTCNN(period=5)
# Create an inception resnet (in eval mode):
verifier = embedResnet(period=15)

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

wait = .1
th1 = .7
th2 = .9
i = 0
fig = plt.figure()
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
video = cv2.VideoWriter('res.avi',fourcc,10,(w,w+h))
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
while True:
    time.sleep(wait)
    ret,frame = vid.read()
    frame = cv2.resize(frame,(w,h))
    output = frame.copy()
    res    = frame.copy()
    verify_status = 'NOT VERIFIED'
    if ret:
        # print(f'{i}-frame, time:{stamp[-1]}s')
        img = Image.fromarray(frame)
        i += 1
        # detection
        try:
            img_cropped, boxes = detector.schedule(img)
        except Exception as e:
            traceback.print_exc()
            print("error:",e)
            print('\n')
            output = np.vstack((output,buffer))
            video.write(output)
            cv2.imshow('res',output)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
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
            
            if norm.min() < th1 and norm.mean() < th2 :
                verify = True
            else:
                verify = False
        else:
                last_verify = verify
        # visualize cpu usage
        plt.plot(stamp[-ss:],cpu_hist[-ss:],linewidth=1,alpha=0.2)
        plt.plot(stamp[-ss:],cpu_ma1[-ss:],linewidth=1,c='black',alpha=0.7,linestyle='--')
        plt.plot(stamp[-ss:],cpu_ma2[-ss:],linewidth=2,c='#38a7ab',
                 path_effects=[pe.Stroke(linewidth=2, foreground='g'), pe.Normal()])
        plt.title('CPU Usage')
        plt.savefig('buffer.jpg')
        
        #visualize webcam and verification
        if last_verify is not None:
            if verify == True:
                verify_status = f'VERIFIED min:{norm_min}, mean:{norm_mean}'
                color = (0,255,255)
            elif verify == False:
                verify_status = f'NOT VERIFIED min:{norm_min}, mean:{norm_mean}'
                color = (0,0,255)
                
        else:   
            if verify == True:
                verify_status = f'VERIFIED min:{norm_min}, mean:{norm_mean}'
                color = (0,255,0)
                
            elif verify == False:
                verify_status = f'NOT VERIFIED min:{norm_min}, mean:{norm_mean}'
                color = (0,0,255)
        if boxes is not None:
            x1,y1,x2,y2 = np.array(boxes[0]).astype(int)
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

        buffer = cv2.imread('buffer.jpg')
        buffer = cv2.resize(buffer,(w,w))
        output = np.vstack((res,buffer))
        video.write(output)
        cv2.imshow('res',output)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        # plt.grid()
        # plt.show()
        # wait and clear
        # plt.pause(0.05)
        
        fig.clear()
          
    else:
        break
plt.close()
cv2.destroyAllWindows()
vid.release()
stop_threads = True



