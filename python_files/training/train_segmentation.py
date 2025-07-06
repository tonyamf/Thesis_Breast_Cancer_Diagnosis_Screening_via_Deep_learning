import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import cv2

from models.unet import UNet
from utils.preprocessing import MammographyPreprocessor
from utils.augmentation import MammographyAugmentation
from utils.metrics import SegmentationMetrics, TensorFlowMetrics
from utils.visualization import ResultVisualizer

class SegmentationTrainer:
    """
    Comprehensive training pipeline for mammography segmentation using U-Net
    """
    
    def __init__(self, config=None):
        """
        Initialize the segmentation trainer
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.preprocessor = MammographyPreprocessor()
        self.augmentation = MammographyAugmentation(
            augmentation_probability=self.config['augmentation_prob']
        )
        self.metrics_calculator = SegmentationMetrics()
        self.visualizer = ResultVisualizer()
        
        # Create directories
        self._create_directories()
    
    def _get_default_config(self):
        """Get default training configuration"""
        return {
            'input_shape': (512, 512, 1),
            'num_classes': 1,  # Binary segmentation
            'filters': 64,
            'learning_rate': 1e-4,
            'batch_size': 8,  # Smaller batch size for memory efficiency
            'epochs': 50,
            'validation_split': 0.2,
            'test_split': 0.15,
            'augmentation_prob': 0.8,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 7,
            'checkpoint_dir': 'checkpoints/segmentation',
            'log_dir': 'logs/segmentation',
            'results_dir': 'results/segmentation',
            'dice_threshold': 0.5
        }
    
    def _create_directories(self):
        """Create necessary directories for saving results"""
        directories = [
            self.config['checkpoint_dir'],
            self.config['log_dir'],
            self.config['results_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def build_model(self):
        """Build and compile the U-Net model"""
        self.model = UNet(
            input_shape=self.config['input_shape'],
            num_classes=self.config['num_classes'],
            filters=self.config['filters']
        )
        
        # Compile with dice loss and custom metrics
        self.model.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=self.model.dice_loss,
            metrics=[
                self.model.dice_coefficient,
                TensorFlowMetrics.iou_score_tf,
                TensorFlowMetrics.precision_tf,
                TensorFlowMetrics.recall_tf,
                'binary_accuracy'
            ]
        )
        
        print("U-Net model built and compiled successfully!")
        print(f"Total parameters: {self.model.model.count_params():,}")
        
        return self.model
    
    def prepare_data(self, images, masks, preprocess=True, augment=True):
        """
        Prepare data for training
        
        Args:
            images: Array of input images
            masks: Array of corresponding segmentation masks
            preprocess: Whether to apply preprocessing
            augment: Whether to apply data augmentation
            
        Returns:
            Prepared datasets (train, validation, test)
        """
        print("Preparing segmentation data...")
        
        # Preprocess images and masks if requested
        if preprocess:
            print("Applying preprocessing...")
            processed_images = []
            processed_masks = []
            
            for img, mask in zip(images, masks):
                # Preprocess image
                processed_img = self.preprocessor.preprocess_image(
                    img,
                    target_size=self.config['input_shape'][:2],
                    apply_normalization=True
                )
                
                # Preprocess mask
                processed_mask = self._preprocess_mask(mask, self.config['input_shape'][:2])
                
                processed_images.append(processed_img)
                processed_masks.append(processed_mask)
            
            images = np.array(processed_images)
            masks = np.array(processed_masks)
        
        # Ensure correct shape
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        if len(masks.shape) == 3:
            masks = np.expand_dims(masks, axis=-1)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, masks,
            test_size=self.config['test_split'],
            random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config['validation_split'] / (1 - self.config['test_split']),
            random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Data augmentation for training set
        if augment:
            print("Applying data augmentation...")
            augmented_images, augmented_masks = self._augment_segmentation_data(
                X_train, y_train
            )
            X_train = np.concatenate([X_train, augmented_images], axis=0)
            y_train = np.concatenate([y_train, augmented_masks], axis=0)
            print(f"Augmented training samples: {len(X_train)}")
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(self.config['batch_size']).shuffle(1000)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(self.config['batch_size'])
        
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(self.config['batch_size'])
        
        # Store for later use
        self.test_data = (X_test, y_test)
        
        return train_dataset, val_dataset, test_dataset
    
    def _preprocess_mask(self, mask, target_size):
        """
        Preprocess segmentation mask
        
        Args:
            mask: Input segmentation mask
            target_size: Target size tuple
            
        Returns:
            Preprocessed mask
        """
        # Resize mask
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Ensure binary values
        mask_binary = (mask_resized > 0.5).astype(np.float32)
        
        return mask_binary
    
    def _augment_segmentation_data(self, images, masks):
        """
        Apply augmentation to segmentation data
        
        Args:
            images: Array of images
            masks: Array of masks
            
        Returns:
            Augmented images and masks
        """
        augmented_images = []
        augmented_masks = []
        
        for img, mask in zip(images, masks):
            # Remove extra dimensions for augmentation
            img_2d = np.squeeze(img)
            mask_2d = np.squeeze(mask)
            
            # Apply augmentation
            aug_img, aug_mask = self.augmentation.apply_albumentations(img_2d, mask_2d)
            
            # Restore dimensions
            aug_img = np.expand_dims(aug_img, axis=-1)
            aug_mask = np.expand_dims(aug_mask, axis=-1)
            
            # Normalize augmented image
            if aug_img.max() > 1.0:
                aug_img = aug_img.astype(np.float32) / 255.0
            
            # Ensure binary mask
            aug_mask = (aug_mask > 0.5).astype(np.float32)
            
            augmented_images.append(aug_img)
            augmented_masks.append(aug_mask)
        
        return np.array(augmented_images), np.array(augmented_masks)
    
    def create_callbacks(self):
        """Create training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks_list = [
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    self.config['checkpoint_dir'],
                    f'best_unet_{timestamp}.h5'
                ),
                monitor='val_dice_coefficient',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=os.path.join(self.config['log_dir'], timestamp),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            
            # Custom callback for saving sample predictions
            SegmentationVisualizationCallback(
                validation_data=None,  # Will be set during training
                save_dir=os.path.join(self.config['results_dir'], 'predictions'),
                frequency=5
            )
        ]
        
        return callbacks_list
    
    def train(self, train_dataset, val_dataset, use_mixed_precision=False):
        """
        Train the segmentation model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            use_mixed_precision: Whether to use mixed precision training
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("Starting U-Net training...")
        
        # Enable mixed precision if requested
        if use_mixed_precision:
            print("Enabling mixed precision training...")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Create callbacks
        training_callbacks = self.create_callbacks()
        
        # Add validation data to visualization callback
        for callback in training_callbacks:
            if isinstance(callback, SegmentationVisualizationCallback):
                callback.validation_data = val_dataset
        
        # Train the model
        history = self.model.train(
            train_dataset,
            val_dataset,
            epochs=self.config['epochs'],
            callbacks=training_callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
        return history
    
    def evaluate_model(self, test_dataset, save_results=True):
        """
        Evaluate the trained model on test data
        
        Args:
            test_dataset: Test dataset
            save_results: Whether to save evaluation results
            
        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        print("Evaluating segmentation model...")
        
        # Get predictions
        predictions = self.model.model.predict(test_dataset)
        
        # Calculate metrics for each test sample
        all_metrics = []
        X_test, y_test = self.test_data
        
        for i in range(len(X_test)):
            y_true = y_test[i]
            y_pred = predictions[i]
            
            # Apply threshold
            y_pred_binary = (y_pred > self.config['dice_threshold']).astype(np.float32)
            
            # Calculate metrics
            sample_metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred_binary)
            all_metrics.append(sample_metrics)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'hausdorff_distance':  # Skip None values
                values = [m[key] for m in all_metrics if m[key] is not None]
                if values:
                    aggregated_metrics[f'mean_{key}'] = np.mean(values)
                    aggregated_metrics[f'std_{key}'] = np.std(values)
        
        # Print results
        print("\nSegmentation Evaluation Results:")
        print(f"Mean Dice Coefficient: {aggregated_metrics['mean_dice_coefficient']:.4f} ± {aggregated_metrics['std_dice_coefficient']:.4f}")
        print(f"Mean IoU Score: {aggregated_metrics['mean_iou_score']:.4f} ± {aggregated_metrics['std_iou_score']:.4f}")
        print(f"Mean Sensitivity: {aggregated_metrics['mean_sensitivity']:.4f} ± {aggregated_metrics['std_sensitivity']:.4f}")
        print(f"Mean Specificity: {aggregated_metrics['mean_specificity']:.4f} ± {aggregated_metrics['std_specificity']:.4f}")
        print(f"Mean Pixel Accuracy: {aggregated_metrics['mean_pixel_accuracy']:.4f} ± {aggregated_metrics['std_pixel_accuracy']:.4f}")
        
        # Visualizations
        if save_results:
            self._save_segmentation_results(aggregated_metrics, X_test, y_test, predictions)
        
        return aggregated_metrics
    
    def _save_segmentation_results(self, metrics, X_test, y_test, predictions):
        """Save segmentation evaluation results and visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save sample predictions
        n_samples = min(10, len(X_test))
        for i in range(n_samples):
            fig = self.visualizer.plot_segmentation_results(
                np.squeeze(X_test[i]),
                np.squeeze(y_test[i]),
                np.squeeze(predictions[i] > self.config['dice_threshold']),
                title=f'Sample {i+1} Segmentation Results'
            )
            
            fig.savefig(
                os.path.join(self.config['results_dir'], f'segmentation_sample_{i+1}_{timestamp}.png'),
                dpi=300, bbox_inches='tight'
            )
            plt.close(fig)
        
        # Save metrics to file
        import json
        with open(
            os.path.join(self.config['results_dir'], f'segmentation_metrics_{timestamp}.json'),
            'w'
        ) as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Segmentation results saved to {self.config['results_dir']}")
    
    def predict_segmentation(self, image, preprocess=True):
        """
        Predict segmentation mask for a single image
        
        Args:
            image: Input image
            preprocess: Whether to preprocess the image
            
        Returns:
            Segmentation prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Preprocess if needed
        if preprocess:
            processed_image = self.preprocessor.preprocess_image(
                image,
                target_size=self.config['input_shape'][:2],
                apply_normalization=True
            )
        else:
            processed_image = image
        
        # Get prediction
        prediction_mask = self.model.predict(processed_image)
        
        # Apply threshold
        binary_mask = (prediction_mask > self.config['dice_threshold']).astype(np.uint8)
        
        return {
            'raw_prediction': prediction_mask,
            'binary_mask': binary_mask,
            'confidence_score': np.mean(prediction_mask[prediction_mask > self.config['dice_threshold']])
        }
    
    def load_model(self, model_path):
        """
        Load a trained segmentation model
        
        Args:
            model_path: Path to the saved model
        """
        if self.model is None:
            self.build_model()
        
        self.model.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'dice_loss': self.model.dice_loss,
                'dice_coefficient': self.model.dice_coefficient,
                'iou_score_tf': TensorFlowMetrics.iou_score_tf,
                'precision_tf': TensorFlowMetrics.precision_tf,
                'recall_tf': TensorFlowMetrics.recall_tf
            }
        )
        print(f"Segmentation model loaded from {model_path}")
    
    def save_model(self, model_path):
        """
        Save the trained segmentation model
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.model.save(model_path)
        print(f"Segmentation model saved to {model_path}")
    
    def get_training_summary(self):
        """
        Get a summary of the training configuration
        
        Returns:
            Training summary dictionary
        """
        return {
            'model_architecture': 'U-Net',
            'input_shape': self.config['input_shape'],
            'num_classes': self.config['num_classes'],
            'filters': self.config['filters'],
            'batch_size': self.config['batch_size'],
            'learning_rate': self.config['learning_rate'],
            'epochs': self.config['epochs'],
            'dice_threshold': self.config['dice_threshold'],
            'augmentation_probability': self.config['augmentation_prob']
        }


class SegmentationVisualizationCallback(callbacks.Callback):
    """
    Custom callback to visualize segmentation predictions during training
    """
    
    def __init__(self, validation_data, save_dir, frequency=5):
        """
        Initialize the visualization callback
        
        Args:
            validation_data: Validation dataset
            save_dir: Directory to save visualizations
            frequency: How often to save visualizations (in epochs)
        """
        super().__init__()
        self.validation_data = validation_data
        self.save_dir = save_dir
        self.frequency = frequency
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        """Save visualization at the end of specified epochs"""
        if (epoch + 1) % self.frequency == 0 and self.validation_data is not None:
            # Get a batch from validation data
            for batch in self.validation_data.take(1):
                images, masks = batch
                
                # Predict
                predictions = self.model.predict(images)
                
                # Save first sample from batch
                self._save_prediction_sample(
                    images[0].numpy(),
                    masks[0].numpy(),
                    predictions[0],
                    epoch + 1
                )
                break
    
    def _save_prediction_sample(self, image, true_mask, pred_mask, epoch):
        """Save a single prediction sample"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(np.squeeze(image), cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # True mask
        axes[1].imshow(np.squeeze(true_mask), cmap='hot')
        axes[1].set_title('True Mask')
        axes[1].axis('off')
        
        # Predicted mask (raw)
        axes[2].imshow(np.squeeze(pred_mask), cmap='hot')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        # Thresholded prediction
        binary_pred = (pred_mask > 0.5).astype(np.float32)
        axes[3].imshow(np.squeeze(binary_pred), cmap='hot')
        axes[3].set_title('Thresholded Prediction')
        axes[3].axis('off')
        
        plt.suptitle(f'Epoch {epoch} - Segmentation Results')
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.save_dir, f'epoch_{epoch:03d}_prediction.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

