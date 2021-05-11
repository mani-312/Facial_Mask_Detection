from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import keras
import tensorflow as tf
import numpy as np
from numpy import array
import cv2
from mediapipe_face_detect import face_detect

# Detect faces predict whether faces masked or not
def detect_and_predict_mask(frame,maskNet):
        boxes = face_detect(frame)
        faces = []
        locs = []
        preds =[]
        if boxes is None:
                return (locs,preds)
        for box in boxes:
                top,right,bottom,left = box
                face = frame[top:bottom,left:right]
                #cv2.imshow("df",face);
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append([left,top,right,bottom])
        if(len(faces)>0):
                preds = maskNet.predict(array(faces))
        return (locs,preds)

# Loading trained maskNet model
maskNet = tf.keras.models.load_model(r"D:\Mask_detection\mask_detector_own\saved_models\Augmented_CNN.model")

frame = cv2.imread( r"C:\Users\MANIKANTA\Desktop\2.jpg")
(locs, preds) = detect_and_predict_mask(frame, maskNet)

# loop over the detected face locations and their corresponding
# locations
for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
	(startX, startY, endX, endY) = box
	(withoutMask, mask) = pred

	# determine the class label and color we'll use to draw
	# the bounding box and text
	label = "Mask" if mask > withoutMask else "No Mask"
	color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

	# include the probability in the label
	label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

	# display the label and bounding box rectangle on the output
	# frame
	cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
	cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
cv2.imshow("Frame", frame)
