# Facial_Mask_Detection
COVID-19 has created a disastrous moment for everyone. We need to fight back to completely eradicate the disease. Several statistical and biological reports suggest that maintaining social distance and covering the faces with masks can help to prevent the spread of COVID-19. So we need to make sure everyone is putting up the mask. We can design a system that can automatically detect the faces and confirm whether each face is masked or not. Correspondingly, if a person is not wearing a mask then respective action will be taken.  

### Dataset details:
You can download the **Dataset** used for training at https://drive.google.com/drive/folders/1NjTxxQKq1QgJMkGhe64l-Or-HhZ8vOin?usp=sharing.

### Model Workflow:
![alt text](https://github.com/mani-312/Facial_Mask_Detection/blob/main/model_flow.png?raw=true)

### Face detection:
MediaPipe face detection was leveraged to detect faces with localization in an image.
For more details about MediaPipe : https://google.github.io/mediapipe/solutions/face_mesh.html

### Model Summary
- Augmentation technique was used to generate images with different lighting, orientation to make the model generalized to all conditions.
- Extracted only the region of face from image and for training the model
- Model architecture involves several layers where each CNN layer is followed by a MaxPooling layer

### Accuracy
- Train_Accuracy touched **96%** and 
- Test_Accuracy : **95%**
     

### Deployed the model into local server
#### Demo
![alt text](https://github.com/mani-312/Facial_Mask_Detection/blob/main/demo.png?raw=true)
