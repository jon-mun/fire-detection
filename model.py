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
        return img, preds_val_t[0, :, :, :]
    
    def evaluate(self, test_images, test_masks):
        # Initialize empty lists to store predictions and ground truth masks
        predictions = []
        ground_truths = []

        # Iterate over test images and masks
        for image_path, mask_path in zip(test_images, test_masks):
            # Load image and ground truth mask
            image = cv2.imread(image_path)
            ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Perform prediction
            _, predicted_mask = self.predict(image_path)

            # Resize predicted mask to match the ground truth mask size
            predicted_mask = cv2.resize(predicted_mask, (ground_truth_mask.shape[1], ground_truth_mask.shape[0]))

            # Append predicted and ground truth masks to the respective lists
            predictions.append(predicted_mask.flatten())
            ground_truths.append(ground_truth_mask.flatten())

        # Convert lists to numpy arrays
        predictions = np.vstack(predictions)
        ground_truths = np.vstack(ground_truths)

        # Flatten masks into 1D arrays
        predictions_flat = predictions.flatten()
        ground_truths_flat = ground_truths.flatten()

        # Calculate evaluation metrics
        intersection = np.logical_and(predictions_flat, ground_truths_flat).sum()
        union = np.logical_or(predictions_flat, ground_truths_flat).sum()
        iou = intersection / union

        # Compute confusion matrix and classification report
        confusion = confusion_matrix(ground_truths_flat, predictions_flat)
        report = classification_report(ground_truths_flat, predictions_flat, target_names=["Background", "Fire"])

        # Print evaluation metrics, confusion matrix, and classification report
        print("Intersection over Union (IoU):", iou)
        print("\nConfusion Matrix:")
        print(confusion)
        print("\nClassification Report:")
        print(report)