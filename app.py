import math
import time

from controllers.slide_controller import *
from controllers.scroll_controller import *
from controllers.zoom_controller import *


import cv2
from hand_tracking.detector import HandDetector
from hand_tracking.gestures import GestureRecognizer



cap = cv2.VideoCapture(0)

detector = HandDetector()
recognizer = GestureRecognizer()

smooth_x = 0
smooth_y = 0
SMOOTHING = 10

last_action_time = 0
ACTION_DELAY = 1.5

PINCH_CLOSE = 25
PINCH_OPEN = 150

last_swipe_time = 0
SWIPE_DELAY = 1

last_zoom_time = 0
ZOOM_DELAY = 1.5


gesture = "No Hand"


while True:
    current_time = time.time()
    success, img = cap.read()

    if not success:
        break

    img = detector.findHands(img)

    lmList = detector.findPosition(img)
    if len(lmList) != 0:

        gesture = recognizer.detect_gesture(lmList)

        current_x = lmList[8]["x"]   # Index finger tip
        thumb_x = lmList[4]["x"]
        thumb_y = lmList[4]["y"]

        index_x = lmList[8]["x"]
        index_y = lmList[8]["y"]

        distance = math.sqrt(
            (index_x - thumb_x) ** 2 +
            (index_y - thumb_y) ** 2
        )
    

        
        current_time = time.time()

        if gesture == "Zoom Gesture":

            if current_time - last_zoom_time > ZOOM_DELAY:

                print("Zoom In")
                zoom_in()

                last_zoom_time = current_time
                                        

           # print("Pinch Distance =", int(distance))

        elif gesture == "Thumbs Up":

            if current_time - last_zoom_time > ZOOM_DELAY:

            
                print("Zoom Out")
                zoom_out()

                last_zoom_time = current_time

    
        

   

    if len(lmList) != 0:
        #print("Index Tip:", lmList[8])
        #print("Index PIP:", lmList[6])

        
        if gesture == "Index Finger Up":

            x = lmList[8]["x"]
            y = lmList[8]["y"]

            screen_width, screen_height = pyautogui.size()

            mouse_x = int(x * screen_width / 640)
            mouse_y = int(y * screen_height / 480)

            smooth_x = smooth_x + (mouse_x - smooth_x) / SMOOTHING
            smooth_y = smooth_y + (mouse_y - smooth_y) / SMOOTHING

            pyautogui.moveTo(int(smooth_x), int(smooth_y))


        

        if current_time - last_action_time > ACTION_DELAY:

            
            if gesture == "Three Fingers Up":

                scroll_down()
                last_action_time = current_time

            elif gesture == "Two Fingers Up":

                scroll_up()
                last_action_time = current_time

        #print(gesture)

    current_time = time.time()

    if current_time - last_action_time > ACTION_DELAY:

        if gesture == "Open Palm":
            next_slide()
            last_action_time = current_time

        elif gesture == "Closed Fist":
            previous_slide()
            last_action_time = current_time

    cv2.putText(
        img,
        gesture,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()