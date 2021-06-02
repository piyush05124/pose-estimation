import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
d=open('face_points.txt','w')
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255,145,0))
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    img=np.zeros((image.shape[0],image.shape[1],3),dtype='uint8')


    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

 
    results = face_mesh.process(image)
##    try:
##      for i,j in enumerate(results.multi_face_landmarks):
##        ih,iw,ic=image.shape
##        print(j.x+iw,j.y+ih)
##    except TypeError:
##      pass

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(image=img,  landmark_list=face_landmarks,  connections=mp_face_mesh.FACE_CONNECTIONS,
                                  landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
        for lm in face_landmarks.landmark:
          d.write(str((lm.x,lm.y)))
    
   
    
       

    cv2.imshow(' Face', image)
    cv2.imshow(' FaceMesh', img)










    
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()


