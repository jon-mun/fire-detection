from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class FireModel:
    def __init__(self, yolo_model_path='models/yolo_fire.pt', unet_model_path='models/unet_fire_best.h5') -> None:
        self.yolo_model = YOLO(yolo_model_path)
        self.unet_model = tf.keras.models.load_model(unet_model_path)
        
    def predict(self, img_path):
        # read image
        img = plt.imread(img_path)
        
        # detect fire using yolo
        detections = self.yolo_model(img_path)
        boxes = detections[0].boxes.data
        
        # isolate/mask fire detection result
        isolated = np.zeros(img.shape, dtype=np.uint8)
        
        for box in boxes:
            isolated[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        
        # segment fire using unet
        isolated = tf.image.resize(isolated, (128, 128))
        preds_val = self.unet_model.predict(np.expand_dims(isolated, axis=0))
        preds_val_t = (preds_val > 0.5).astype(np.uint8)
        
        # return original image and fire segmentation result
        return img, preds_val_t