import cv2
import mediapipe as mp
import math
# Assuming detector.py is in the same directory or accessible via PYTHONPATH
# This import is needed for the example usage block (if __name__ == "__main__")
# In a real application, you would import HandDetector from its specific path.
try:
    from hand_tracking.detector import HandDetector
except ImportError:
    print("Could not import HandDetector. Ensure detector.py is accessible for example usage.")
    # Define a dummy HandDetector if it's not available, to allow the class definition to be parsed
    class HandDetector:
        def __init__(self, *args, **kwargs):
            pass
        def findHands(self, img, draw=True):
            return img
        def findPosition(self, img, hand_no=0, draw=True):
            return []

"""
This module provides a GestureRecognizer class for detecting common hand gestures
based on MediaPipe hand landmark positions.
"""

class GestureRecognizer:
    """
    Recognizes hand gestures (Open Palm, Closed Fist, Index Finger Up ,Two Fingers Up, Three Fingers Up)
    from a list of hand landmark coordinates provided by a hand detector.
    """
    def __init__(self):
        # Landmark IDs for the tips of the fingers
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20

        # Landmark IDs for the PIP (Proximal Interphalangeal) joints
        # For the thumb, we use the IP (Interphalangeal) joint as it's the closest
        # equivalent for detecting bending.
        self.THUMB_IP = 3
        self.INDEX_PIP = 6
        self.MIDDLE_PIP = 10
        self.RING_PIP = 14
        self.PINKY_PIP = 18

        # Landmark IDs for the MCP (Metacarpophalangeal) joints
        self.THUMB_MCP = 2
        self.INDEX_MCP = 5
        self.MIDDLE_MCP = 9
        self.RING_MCP = 13
        self.PINKY_MCP = 17

        # A small threshold in pixels to account for minor variations in landmark detection
        # and to make the detection more robust to slight hand movements.
        self.Y_THRESHOLD = 15 # Adjust this value as needed for sensitivity

    def _is_finger_extended(self, lm_list, tip_id, pip_id):
        """
        Checks if a non-thumb finger (Index, Middle, Ring, Pinky) is extended.
        A finger is considered extended if its tip's y-coordinate is significantly
        above its PIP joint's y-coordinate (assuming the hand is oriented upright).
        """
        if not lm_list or len(lm_list) < max(tip_id, pip_id) + 1:
            return False
        # If the tip's y-coordinate is less than the PIP's y-coordinate (higher on the screen),
        # the finger is extended. We subtract a threshold for robustness.
        return lm_list[tip_id]['y'] < lm_list[pip_id]['y'] - self.Y_THRESHOLD

    def _is_thumb_extended(self, lm_list):
        """
        Checks if the thumb is extended.
        For the thumb, we compare the y-coordinate of the tip with its MCP joint.
        This assumes a relatively upright hand orientation.
        """
        if not lm_list or len(lm_list) < max(self.THUMB_TIP, self.THUMB_MCP) + 1:
            return False
        # If the thumb tip's y-coordinate is less than its MCP's y-coordinate (higher on the screen),
        # the thumb is extended. We subtract a threshold for robustness.
        return lm_list[self.THUMB_TIP]['y'] < lm_list[self.THUMB_MCP]['y'] - self.Y_THRESHOLD

    def _is_open_palm(self, lm_list):
        """
        Detects if the hand is in an 'Open Palm' gesture.
        This gesture is recognized when all fingers (thumb, index, middle, ring, pinky)
        are extended.
        """
        if not lm_list:
            return False

        # Check if each finger is extended
        thumb_extended = self._is_thumb_extended(lm_list)
        index_extended = self._is_finger_extended(lm_list, self.INDEX_TIP, self.INDEX_PIP)
        middle_extended = self._is_finger_extended(lm_list, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_extended = self._is_finger_extended(lm_list, self.RING_TIP, self.RING_PIP)
        pinky_extended = self._is_finger_extended(lm_list, self.PINKY_TIP, self.PINKY_PIP)

        # All fingers must be extended for an open palm
        return thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended

    def _is_closed_fist(self, lm_list):

        fingers = [
            (self.INDEX_TIP, self.INDEX_PIP),
            (self.MIDDLE_TIP, self.MIDDLE_PIP),
            (self.RING_TIP, self.RING_PIP),
            (self.PINKY_TIP, self.PINKY_PIP)
        ]

        for tip, pip in fingers:
            if lm_list[tip]['y'] < lm_list[pip]['y']:
                return False

        return True

    def _is_index_finger_up(self, lm_list):

        index_extended = self._is_finger_extended(
            lm_list,
            self.INDEX_TIP,
            self.INDEX_PIP
        )

        middle_extended = self._is_finger_extended(
            lm_list,
            self.MIDDLE_TIP,
            self.MIDDLE_PIP
        )

        ring_extended = self._is_finger_extended(
            lm_list,
            self.RING_TIP,
            self.RING_PIP
        )

        pinky_extended = self._is_finger_extended(
            lm_list,
            self.PINKY_TIP,
            self.PINKY_PIP
        )
        print(
            f"Index={index_extended}, "
            f"Middle={middle_extended}, "
            f"Ring={ring_extended}, "
            f"Pinky={pinky_extended}"
        )

        return (
            index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
        )
    def _is_two_fingers_up(self, lm_list):

        index_extended = self._is_finger_extended(
            lm_list,
            self.INDEX_TIP,
            self.INDEX_PIP
        )

        middle_extended = self._is_finger_extended(
            lm_list,
            self.MIDDLE_TIP,
            self.MIDDLE_PIP
        )

        ring_bent = (
            lm_list[self.RING_TIP]["y"]
            > lm_list[self.RING_PIP]["y"] + self.Y_THRESHOLD
        )

        pinky_bent = (
            lm_list[self.PINKY_TIP]["y"]
            > lm_list[self.PINKY_PIP]["y"] + self.Y_THRESHOLD
        )

        return (
            index_extended
            and middle_extended
            and ring_bent
            and pinky_bent
        )
    
    def _is_three_fingers_up(self, lm_list):

        index_extended = self._is_finger_extended(
            lm_list, self.INDEX_TIP, self.INDEX_PIP
        )

        middle_extended = self._is_finger_extended(
            lm_list, self.MIDDLE_TIP, self.MIDDLE_PIP
        )

        ring_extended = self._is_finger_extended(
            lm_list, self.RING_TIP, self.RING_PIP
        )

        pinky_bent = (
            lm_list[self.PINKY_TIP]["y"]
            > lm_list[self.PINKY_PIP]["y"]
        )

        return (
            index_extended
            and middle_extended
            and ring_extended
            and pinky_bent
        )
    
    def _is_thumbs_up(self, lm_list):

        thumb_extended = self._is_thumb_extended(lm_list)

        index_bent = not self._is_finger_extended(
            lm_list, self.INDEX_TIP, self.INDEX_PIP
        )

        middle_bent = not self._is_finger_extended(
            lm_list, self.MIDDLE_TIP, self.MIDDLE_PIP
        )

        ring_bent = not self._is_finger_extended(
            lm_list, self.RING_TIP, self.RING_PIP
        )

        pinky_bent = not self._is_finger_extended(
            lm_list, self.PINKY_TIP, self.PINKY_PIP
        )

        return (
            thumb_extended
            and index_bent
            and middle_bent
            and ring_bent
            and pinky_bent
        )

    def detect_gesture(self, lm_list):

        if not lm_list or len(lm_list) < 21:
            return "No Hand Detected"

        thumb_x = lm_list[self.THUMB_TIP]["x"]
        thumb_y = lm_list[self.THUMB_TIP]["y"]

        index_x = lm_list[self.INDEX_TIP]["x"]
        index_y = lm_list[self.INDEX_TIP]["y"]

        pinch_distance = math.sqrt(
            (index_x - thumb_x) ** 2 +
            (index_y - thumb_y) ** 2
        )

        
        # Detect normal gestures first
        if self._is_open_palm(lm_list):
            return "Open Palm"

        elif self._is_closed_fist(lm_list):
            return "Closed Fist"

        elif self._is_index_finger_up(lm_list):
            return "Index Finger Up"

        elif self._is_two_fingers_up(lm_list):
            return "Two Fingers Up"

        elif self._is_three_fingers_up(lm_list):
            return "Three Fingers Up"

        elif pinch_distance < 50:
            return "Zoom Gesture"
        
        elif self._is_thumbs_up(lm_list):
            return "Thumbs Up"

        return "Unknown Gesture"
if __name__ == "__main__":
    # This block demonstrates how to use the GestureRecognizer with a webcam feed.
    # It requires the HandDetector class from detector.py to be available.
    cap = cv2.VideoCapture(0)  # Open the default webcam
    detector = HandDetector()  # Create an instance of the HandDetector
    recognizer = GestureRecognizer() # Create an instance of the GestureRecognizer

    while True:
        success, img = cap.read()  # Read a frame from the webcam
        if not success:
            print("Failed to grab frame")
            break

        # Find hands in the frame and draw landmarks
        img = detector.findHands(img)
        # Get the position of all landmarks for the first detected hand
        lm_list = detector.findPosition(img)

        gesture = "No Hand Detected"
        if lm_list:
            # Detect the gesture using the landmark list
            gesture = recognizer.detect_gesture(lm_list)

        # Display the detected gesture on the image
        cv2.putText(img, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Gesture Recognizer", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()