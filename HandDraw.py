#---------------------------------------
# Install:
#    pip install mediapipe
# Usage:
#    Press b to draw
#    Press r to delete
#    Press q to exit
#---------------------------------------

import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
from typing import List, Optional, Tuple, Union
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or 
            math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
        is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
        return None

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

# Take the 3D np array cache
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, cache = cap.read()
    break

# Take the image size
image_rows, image_cols, _ = cache.shape

# Convert cache to black
cache *= 0

# Initialize some variables
previous_point = (-1,-1)
track_started = False


with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        cv2.waitKey(1)

        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # Copy cache to image
        image = cache.copy()

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                # Get the landmark coordinates
                idx_to_coordinates = {}
                for idx, landmark in enumerate(hand.landmark):
                    if ((landmark.HasField('visibility') and
                        landmark.visibility < VISIBILITY_THRESHOLD) or
                        (landmark.HasField('presence') and
                        landmark.presence < PRESENCE_THRESHOLD)):
                        continue
                    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, 
                        image_cols, image_rows)   
                    if landmark_px:
                        idx_to_coordinates[idx] = landmark_px 
                
                # Draw the stroke to cache
                if 8 in idx_to_coordinates:
                    if track_started:
                        cache = cv2.circle(cache, idx_to_coordinates[8],2,(255,255,255),-1)
                        cache = cv2.line(cache, previous_point, idx_to_coordinates[8],(255,255,255),3)
                    previous_point = idx_to_coordinates[8]

                # Draw the hand to image
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, 
                    circle_radius=4), mp_drawing.DrawingSpec(color=(250, 44, 250), 
                    thickness=2, circle_radius=2),)            

        cv2.imshow('Hand Tracking', image)
        cv2.imshow('Cache', cache)

        key = cv2.waitKey(10) & 0xFF

        # Break if 'q' pressed
        if key == ord('q'):
            break

        # Delete cache if 'q' pressed
        if key == ord('r'):
            cache *= 0

        # Start drawing if 'b' pressed
        if key == ord('b'):
            track_started = True
        else:
            track_started = False

cap.release()
cv2.destroyAllWindows()
