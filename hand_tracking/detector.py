import cv2
import mediapipe as mp

"""
This module provides a HandDetector class for detecting hands,
drawing landmarks, and returning landmark positions using MediaPipe.
"""

class HandDetector:
    """
    Detects hands in an image using MediaPipe and provides methods
    to find hand landmarks and their positions.
    """
    def __init__(self, mode=False, max_num_hands=1, model_complexity=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initializes the HandDetector with MediaPipe Hands parameters.

        Args:
            mode (bool): If set to false, the solution treats the input images as a video stream.
                         It will try to detect hands in the first input images, and upon successful
                         detection, it will then track the hands in the subsequent images.
                         If set to true, the solution treats the input images as a batch of static
                         and possibly unrelated images. Hand detection runs on every input image.
            max_num_hands (int): Maximum number of hands to detect.
            model_complexity (int): Complexity of the hand landmark model: 0, 1, or 2.
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for hand
                                              detection to be considered successful.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for the
                                             hand landmarks to be considered tracked successfully.
        """
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands
        # Create a Hands object with specified parameters
        self.hands = self.mp_hands.Hands(self.mode, self.max_num_hands, self.model_complexity,
                                         self.min_detection_confidence, self.min_tracking_confidence)
        # Initialize MediaPipe drawing utilities
        self.mp_draw = mp.solutions.drawing_utils
        # Store results from hand processing
        self.results = None

    def findHands(self, img, draw=True):
        """
        Detects hands in the input image and optionally draws the landmarks and connections.

        Args:
            img (numpy.ndarray): The input image (BGR format from OpenCV).
            draw (bool): If True, draws landmarks and connections on the image.

        Returns:
            numpy.ndarray: The image with (or without) drawn landmarks.
        """
        # Convert the BGR image to RGB, as MediaPipe Hands requires RGB input
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the RGB image to find hands
        self.results = self.hands.process(img_rgb)

        # Check if any hands are detected
        if self.results.multi_hand_landmarks:
            # Iterate through each detected hand
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # Draw the landmarks and connections on the original BGR image
                    self.mp_draw.draw_landmarks(img, hand_landmarks,
                                                self.mp_hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, hand_no=0, draw=True):
        """
        Finds the position of all landmarks for a specific hand.

        Args:
            img (numpy.ndarray): The input image.
            hand_no (int): The index of the hand to get landmarks for (e.g., 0 for the first hand).
            draw (bool): If True, draws a circle around a specific landmark (e.g., index finger tip).

        Returns:
            list: A list of dictionaries, where each dictionary contains the id, x, and y
                  coordinates of a landmark. Returns an empty list if no hand is detected.
        """
        lm_list = []
        # Check if hand landmarks were detected in the previous findHands call
        if self.results and self.results.multi_hand_landmarks:
            # Get the landmarks for the specified hand
            my_hand = self.results.multi_hand_landmarks[hand_no]
            # Get image dimensions
            h, w, c = img.shape
            # Iterate through each landmark
            for id, lm in enumerate(my_hand.landmark):
                # Convert normalized landmark coordinates to pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append({"id": id, "x": cx, "y": cy})
                if draw and id == 8:  # Example: Draw a circle around the tip of the index finger (landmark ID 8)
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lm_list

if __name__ == "__main__":
    # Main block for testing the HandDetector class with a webcam feed
    cap = cv2.VideoCapture(0)  # Open the default webcam
    detector = HandDetector()  # Create an instance of the HandDetector

    while True:
        success, img = cap.read()  # Read a frame from the webcam
        if not success:
            print("Failed to grab frame")
            break

        # Find hands in the frame and draw landmarks
        img = detector.findHands(img)
        # Get the position of all landmarks for the first detected hand
        lm_list = detector.findPosition(img)

        # If landmarks are detected, print the position of the index finger tip (landmark ID 8)
        if lm_list:
            # print(f"Index finger tip position: x={lm_list[8]['x']}, y={lm_list[8]['y']}")
            pass # You can add custom logic here based on landmark positions

        # Display the frame
        cv2.imshow("Hand Detector", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()