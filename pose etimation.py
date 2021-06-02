import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)

ptime=0


mpo=mp.solutions.pose
pose=mpo.Pose()
mp_hands = mp.solutions.hands

mpdraw=mp.solutions.drawing_utils
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
      
      while True:
                  ret, image = cap.read()
                  img = np.zeros((image.shape[0],image.shape[1] ,3), dtype = "uint8")
                  rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                  res=pose.process(rgb)
                  results = hands.process(rgb)
                  file=open('points.txt','w')
                  file.write(str(res.pose_landmarks))
                  
                  if res.pose_landmarks:
                        mpdraw.draw_landmarks(img,res.pose_landmarks,mpo.POSE_CONNECTIONS)

                  if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                              mpdraw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)


                  ctime=time.time()
                  fps=1/(ctime-ptime)
                  ptime=ctime

                  










                  cv2.putText(image,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,125),3)
                  cv2.imshow("full", image)
                  cv2.imshow('Single Channel Window', cv2.resize(img,(1080,720)))
                  key = cv2.waitKey(1)
                  if key & 0xFF == ord('q') or key & 0xFF == ord('Q'):
                      break


cap.release()
cv2.destroyAllWindows()
