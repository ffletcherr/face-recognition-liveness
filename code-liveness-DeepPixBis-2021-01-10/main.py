from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import csv
from utils import DeepPixBiSExtractor

model = DeepPixBiSExtractor(scoring_method='combined', #['pixel_mean','binary','combined']
                            model_file='Pretrained_models/OULU_Protocol_2_model_0_0.pth')

       
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (15,35)
fontScale              = 1
lineType               = 2     
color                  = (150,0,100)
wait = .2
w,h = (1280//2,720//2)
vid = cv2.VideoCapture(0)

while True:
    ret,first_frame = vid.read()
    time.sleep(wait)
    if ret:
        break
    else:
        print('no webcam!')
print('get frist frame!')
# first_frame = cv2.resize(first_frame,(w,h))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
h,w = tuple(first_frame.shape[:2])
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', w,h)
orig_video = cv2.VideoWriter('original.avi',fourcc,5,(w,h))
test_video = cv2.VideoWriter('processed.avi',fourcc,5,(w,h))

print('begin real-time process')
i = 0
m1 = 5
m2 = 15
vote_w = 20
live_th = .5
cd = 380 // 2
liveness_hist = []
mtcnn  = MTCNN()
hist_csv = open('history.csv','w')
writer = csv.writer(hist_csv)
while True:
    ret,frame = vid.read()
    # bgr = cv2.resize(frame,(w,h))
    time.sleep(.01)
    if ret:
        bgr = frame.copy()
        orig_video.write(bgr)
        rgb  =  cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        try:
            print('start detection...')
            face, boxes = mtcnn(rgb)
            boxes = [max(qq,0) for qq in list(boxes[0])]
            x1,y1,x2,y2 = np.array(boxes).astype(int)
            tight_crop = rgb[y1:y2,x1:x2]
            
        except Exception as e: 
            print(e)
            continue
        try:
            tight = cv2.resize(tight_crop,(224,224)).transpose(2,0,1)
            liveness_score  =  model(tight)
            liveness_hist.append(liveness_score)
            i += 1
            print(tight_crop.shape,rgb.shape)
            res = cv2.putText(  bgr,f'liveness:{liveness_score:.3f}', 
                                (15,35), 
                                font, 
                                fontScale,
                                color,
                                lineType 
                                )
            mean1 = np.mean(liveness_hist[-m1:])
            res = cv2.putText(  bgr,f'mean1:{mean1:.3f}', 
                                (15,35+30), 
                                font, 
                                fontScale,
                                color,
                                lineType 
                                )
            mean2 = np.mean(liveness_hist[-m2:])
            res = cv2.putText(  bgr,f'mean2:{mean2:.3f}', 
                                (15,35+30+30), 
                                font, 
                                fontScale,
                                color,
                                lineType 
                                )
            voting_count = (np.array(liveness_hist[-vote_w:])>live_th).sum()
            res = cv2.putText(  bgr,f'voting:{voting_count:.3f}', 
                                (15,35+30+30+30), 
                                font, 
                                fontScale,
                                color,
                                lineType 
                                )
            print('\n')
            
            cv2.imshow('frame',res)
            k = cv2.waitKey(1)
            test_video.write(res)
            writer.writerow([liveness_score,mean1,mean2,voting_count])
            if k == ord('q'):
                
                break
            

        except Exception as e: 
                    print(e)
                    continue
    else:
        break
cv2.destroyAllWindows()
hist_csv.close()
vid.release()
test_video.release()
orig_video.release()

