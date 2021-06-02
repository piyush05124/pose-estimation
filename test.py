import cv2
import numpy as np




cap = cv2.VideoCapture(0)
while True:
      _,image=cap.read()
      image=cv2.flip(image,1)
     # print(image.shape[2])
      img = np.zeros((image.shape[0],image.shape[1] ,3), dtype = "uint8")

      


      
  
      cv2.circle(img,(447,63), 2, (255,0,250), 5)
      cv2.imshow('Single Channel Window', img)
      cv2.imshow('MediaPipe Hands', image)
      if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
