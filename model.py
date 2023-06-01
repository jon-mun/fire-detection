from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

class FireModel:
    def __init__(self, yolo_model_path='models/yolo_fire.pt', unet_model_path='models/unet_fire_best.h5') -> None:
        self.yolo_model = YOLO(yolo_model_path)
        self.unet_model = tf.keras.models.load_model(unet_model_path)
        
    def predict(self, img_path, output=True):
        # read image
        img = plt.imread(img_path)
        
        # detect fire using yolo
        detections = self.yolo_model(img_path, verbose=output)
        boxes = detections[0].boxes.data
        
        # isolate/mask fire detection result
        isolated = np.zeros(img.shape, dtype=np.uint8)
        
        for box in boxes:
            isolated[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        
        # segment fire using unet
        verbose = 1
        if not output:
          verbose = 0
        
        isolated = tf.image.resize(isolated, (128, 128))
        preds_val = self.unet_model.predict(np.expand_dims(isolated, axis=0), verbose=verbose)
        preds_val_t = (preds_val > 0.5).astype(np.uint8)

        # return original image and fire segmentation result
        return img, preds_val_t[0, :, :, :]
    
    def evaluate(self, test_images, test_masks, output=False, visualize=False):
        # Initialize empty lists to store evaluation metrics
        predictions = []
        ground_truths = []
        iou_scores = []
        dice_scores = []
        pixel_accuracy_scores = []

        # Iterate over test images and masks
        for image_path, mask_path in zip(test_images, test_masks):
            # Load image and ground truth mask
            image = cv2.imread(image_path)
            ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Perform prediction
            _, predicted_mask = self.predict(image_path, output=output)

            # Resize predicted mask to match the ground truth mask size
            predicted_mask = cv2.resize(predicted_mask, (ground_truth_mask.shape[1], ground_truth_mask.shape[0]))

            # visualize predictions
            if visualize:
              self.visualize_eval(input=_, pred=predicted_mask, gt=ground_truth_mask)

            # flatten and calculate iou
            pred_flat = predicted_mask.flatten()
            gt_flat = ground_truth_mask.flatten()

            # Calculate evaluation metrics
            intersection = np.logical_and(pred_flat, gt_flat).sum()
            union = np.logical_or(pred_flat, gt_flat).sum()
            iou = intersection/union
            dice = (2 * np.sum(intersection)) / (np.sum(predicted_mask) + np.sum(ground_truth_mask))
            pixel_accuracy = np.sum(predicted_mask == ground_truth_mask) / (ground_truth_mask.shape[0] * ground_truth_mask.shape[1])

            # Append evaluation metrics
            predictions.append(predicted_mask.flatten())
            ground_truths.append(ground_truth_mask.flatten())
            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracy_scores.append(pixel_accuracy)

        # Calculate average
        mean_iou = np.mean(iou_scores)
        mean_dice = np.mean(dice_scores)
        mean_pixel_accuracy = np.mean(pixel_accuracy_scores)

        # Print evaluation metrics
        print("Mean Average Intersection over Union (mIoU):", mean_iou)
        print("Mean Dice Coefficient:", mean_dice)
        print("Mean Pixel Accuracy:", mean_pixel_accuracy)

    @staticmethod
    def visualize_eval(input, pred, gt):
        fig, ax = plt.subplots(nrows=1, ncols=3)
              
        ax[0].set_title('input image')
        ax[0].imshow(input)

        ax[1].set_title('prediction')
        ax[1].imshow(pred, cmap='gray')

        ax[2].set_title('ground truth')
        ax[2].imshow(gt, cmap='gray')