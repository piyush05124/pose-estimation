import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands



cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()

    img = np.zeros((image.shape[0],image.shape[1] ,3), dtype = "uint8")
  
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        print(hand_landmarks.INDEX_FINGER_TIP)
        

        
    cv2.imshow('MediaPipe Hands', image)
    cv2.imshow('Single Channel Window', img)
    if cv2.waitKey(1) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
