import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

from models.efficientnet import EfficientNetClassifier
from utils.preprocessing import MammographyPreprocessor
from utils.augmentation import MammographyAugmentation
from utils.metrics import ClassificationMetrics
from utils.visualization import ResultVisualizer

class ClassificationTrainer:
    """
    Comprehensive training pipeline for mammography classification using EfficientNet
    """
    
    def __init__(self, config=None):
        """
        Initialize the classification trainer
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.preprocessor = MammographyPreprocessor()
        self.augmentation = MammographyAugmentation(
            augmentation_probability=self.config['augmentation_prob']
        )
        self.metrics_calculator = ClassificationMetrics(
            num_classes=self.config['num_classes'],
            class_names=self.config['class_names']
        )
        self.visualizer = ResultVisualizer()
        
        # Create directories
        self._create_directories()
    
    def _get_default_config(self):
        """Get default training configuration"""
        return {
            'num_classes': 3,
            'class_names': ['Normal', 'Benign', 'Malignant'],
            'input_shape': (224, 224, 3),
            'efficientnet_version': 'B0',
            'freeze_base': True,
            'dropout_rate': 0.2,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'epochs': 50,
            'validation_split': 0.2,
            'test_split': 0.15,
            'augmentation_prob': 0.8,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'checkpoint_dir': 'checkpoints/classification',
            'log_dir': 'logs/classification',
            'results_dir': 'results/classification'
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
        """Build and compile the EfficientNet model"""
        self.model = EfficientNetClassifier(
            num_classes=self.config['num_classes'],
            input_shape=self.config['input_shape'],
            efficientnet_version=self.config['efficientnet_version'],
            freeze_base=self.config['freeze_base'],
            dropout_rate=self.config['dropout_rate']
        )
        
        # Compile model
        self.model.compile_model(
            optimizer='adam',
            learning_rate=self.config['learning_rate']
        )
        
        print("Model built and compiled successfully!")
        print(f"Total parameters: {self.model.model.count_params():,}")
        
        return self.model
    
    def prepare_data(self, images, labels, preprocess=True, augment=True):
        """
        Prepare data for training
        
        Args:
            images: Array of input images
            labels: Array of corresponding labels
            preprocess: Whether to apply preprocessing
            augment: Whether to apply data augmentation
            
        Returns:
            Prepared datasets (train, validation, test)
        """
        print("Preparing data...")
        
        # Preprocess images if requested
        if preprocess:
            print("Applying preprocessing...")
            processed_images = []
            for img in images:
                processed_img = self.preprocessor.preprocess_image(
                    img,
                    target_size=self.config['input_shape'][:2]
                )
                # Convert grayscale to RGB for EfficientNet
                if len(processed_img.shape) == 2:
                    processed_img = np.stack([processed_img] * 3, axis=-1)
                processed_images.append(processed_img)
            images = np.array(processed_images)
        
        # Convert labels to categorical if needed
        if len(labels.shape) == 1:
            labels = tf.keras.utils.to_categorical(labels, self.config['num_classes'])
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=self.config['test_split'],
            random_state=42,
            stratify=np.argmax(labels, axis=1)
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config['validation_split'] / (1 - self.config['test_split']),
            random_state=42,
            stratify=np.argmax(y_temp, axis=1)
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Data augmentation for training set
        if augment:
            print("Applying data augmentation...")
            augmented_data = self.augmentation.augment_dataset(
                X_train, np.argmax(y_train, axis=1),
                augmentation_factor=2
            )
            X_train = augmented_data['images']
            y_train = tf.keras.utils.to_categorical(
                augmented_data['labels'], self.config['num_classes']
            )
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
    
    def create_callbacks(self):
        """Create training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks_list = [
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    self.config['checkpoint_dir'],
                    f'best_model_{timestamp}.h5'
                ),
                monitor='val_accuracy',
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
                factor=0.2,
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
            
            # Learning rate scheduler
            callbacks.LearningRateScheduler(
                lambda epoch: self.config['learning_rate'] * (0.95 ** epoch),
                verbose=0
            )
        ]
        
        return callbacks_list
    
    def train(self, train_dataset, val_dataset, use_mixed_precision=False):
        """
        Train the classification model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            use_mixed_precision: Whether to use mixed precision training
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("Starting training...")
        
        # Enable mixed precision if requested
        if use_mixed_precision:
            print("Enabling mixed precision training...")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Create callbacks
        training_callbacks = self.create_callbacks()
        
        # Train the model
        history = self.model.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config['epochs'],
            callbacks=training_callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
        return history
    
    def fine_tune_model(self, train_dataset, val_dataset, fine_tune_epochs=10):
        """
        Fine-tune the pre-trained model with unfrozen layers
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            fine_tune_epochs: Number of fine-tuning epochs
            
        Returns:
            Fine-tuning history
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        print("Starting fine-tuning...")
        
        # Unfreeze the base model
        base_model = None
        for layer in self.model.model.layers:
            if 'efficientnet' in layer.name.lower():
                base_model = layer
                break
        
        if base_model:
            base_model.trainable = True
            print(f"Unfrozen {len(base_model.layers)} layers in base model")
        
        # Use a lower learning rate for fine-tuning
        fine_tune_lr = self.config['learning_rate'] / 10
        
        self.model.model.compile(
            optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
            loss=self.model.model.loss,
            metrics=self.model.model.metrics
        )
        
        # Create callbacks for fine-tuning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fine_tune_callbacks = [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    self.config['checkpoint_dir'],
                    f'fine_tuned_model_{timestamp}.h5'
                ),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        # Fine-tune the model
        fine_tune_history = self.model.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=fine_tune_epochs,
            callbacks=fine_tune_callbacks,
            verbose=1
        )
        
        print("Fine-tuning completed!")
        
        return fine_tune_history
    
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
        
        print("Evaluating model...")
        
        # Get predictions
        predictions = self.model.model.predict(test_dataset)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(self.test_data[1], axis=1)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            y_true, y_pred, predictions
        )
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
        
        if 'auc_roc_ovr' in metrics and metrics['auc_roc_ovr'] is not None:
            print(f"AUC-ROC (OvR): {metrics['auc_roc_ovr']:.4f}")
        
        # Per-class metrics
        print("\nPer-class Metrics:")
        for class_name in self.config['class_names']:
            if f'precision_{class_name}' in metrics:
                print(f"{class_name}:")
                print(f"  Precision: {metrics[f'precision_{class_name}']:.4f}")
                print(f"  Recall: {metrics[f'recall_{class_name}']:.4f}")
                print(f"  F1-score: {metrics[f'f1_{class_name}']:.4f}")
        
        # Visualizations
        if save_results:
            self._save_evaluation_results(metrics, y_true, y_pred, predictions)
        
        return metrics
    
    def _save_evaluation_results(self, metrics, y_true, y_pred, predictions):
        """Save evaluation results and visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save confusion matrix
        cm_fig = self.visualizer.plot_confusion_matrix_interactive(
            y_true, y_pred, self.config['class_names']
        )
        cm_fig.write_html(
            os.path.join(self.config['results_dir'], f'confusion_matrix_{timestamp}.html')
        )
        
        # Save ROC curves
        roc_fig = self.visualizer.plot_roc_curves_interactive(
            tf.keras.utils.to_categorical(y_true, self.config['num_classes']),
            predictions,
            self.config['class_names']
        )
        roc_fig.write_html(
            os.path.join(self.config['results_dir'], f'roc_curves_{timestamp}.html')
        )
        
        # Save metrics to file
        import json
        # Convert numpy types to Python types for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            elif isinstance(value, (np.int64, np.float64)):
                json_metrics[key] = value.item()
            elif key != 'classification_report':  # Skip complex nested dict
                json_metrics[key] = value
        
        with open(
            os.path.join(self.config['results_dir'], f'metrics_{timestamp}.json'),
            'w'
        ) as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"Results saved to {self.config['results_dir']}")
    
    def predict_single_image(self, image, preprocess=True):
        """
        Predict class for a single image
        
        Args:
            image: Input image
            preprocess: Whether to preprocess the image
            
        Returns:
            Prediction results dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        # Preprocess if needed
        if preprocess:
            processed_image = self.preprocessor.preprocess_image(
                image,
                target_size=self.config['input_shape'][:2]
            )
            # Convert grayscale to RGB
            if len(processed_image.shape) == 2:
                processed_image = np.stack([processed_image] * 3, axis=-1)
        else:
            processed_image = image
        
        # Get prediction
        prediction_probs = self.model.predict(processed_image)
        predicted_class, confidence = self.model.predict_class(processed_image)
        
        return {
            'predicted_class': self.config['class_names'][predicted_class],
            'confidence': confidence,
            'class_probabilities': {
                name: prob for name, prob in zip(
                    self.config['class_names'], prediction_probs[0]
                )
            }
        }
    
    def load_model(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path: Path to the saved model
        """
        if self.model is None:
            self.build_model()
        
        self.model.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def save_model(self, model_path):
        """
        Save the trained model
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def get_training_summary(self):
        """
        Get a summary of the training configuration
        
        Returns:
            Training summary dictionary
        """
        return {
            'model_architecture': f"EfficientNet{self.config['efficientnet_version']}",
            'num_classes': self.config['num_classes'],
            'input_shape': self.config['input_shape'],
            'batch_size': self.config['batch_size'],
            'learning_rate': self.config['learning_rate'],
            'epochs': self.config['epochs'],
            'augmentation_probability': self.config['augmentation_prob'],
            'class_names': self.config['class_names']
        }

