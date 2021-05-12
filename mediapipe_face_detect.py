import cv2
import mediapipe as mp
import math
from typing import List, Tuple, Union



def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  #if not (is_valid_normalized_value(normalized_x) and
   #       is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    #return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def face_detect(image):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # For static images:
    face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5)

    #image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    #results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    results = face_detection.process(image)
    # Draw face detections of each face.
    if not results.detections:
        return
    annotated_image = image.copy()

    boxes = []
    for detection in results.detections:
        
        image_rows, image_cols, _ = image.shape
        location = detection.location_data
        
        if not location.HasField('relative_bounding_box'):
            continue
        relative_bounding_box = location.relative_bounding_box
        relative_bounding_box = location.relative_bounding_box
        
        left,top = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
              image_rows)
        
        right,bottom = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + +relative_bounding_box.height, image_cols,
            image_rows)

        
        # Send the image coordinates in order as per face_recognition
        # ROI = image[top:bttom,left:right]
        boxes.append([top,right,bottom,left])
        
    return boxes

img_path = r"D:\Face_recognition\test_images\test_mask2.jpg"
image = cv2.imread(img_path)
# Convert the BGR image to RGB and process it with MediaPipe Face Detection.
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the bounding boxes
boxes = face_detect(image)
