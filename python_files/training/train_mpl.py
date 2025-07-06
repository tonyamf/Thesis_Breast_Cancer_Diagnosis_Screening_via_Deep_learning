import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import json

from models.meta_pseudo_labels import MetaPseudoLabels
from models.efficientnet import EfficientNetClassifier
from utils.preprocessing import MammographyPreprocessor
from utils.augmentation import MammographyAugmentation
from utils.metrics import ClassificationMetrics
from utils.visualization import ResultVisualizer

class MPLTrainer:
    """
    Comprehensive training pipeline for Meta Pseudo Labels semi-supervised learning
    """
    
    def __init__(self, config=None):
        """
        Initialize the MPL trainer
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config or self._get_default_config()
        self.mpl_framework = None
        self.teacher_model = None
        self.student_model = None
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
            'teacher_learning_rate': 1e-4,
            'student_learning_rate': 1e-3,
            'batch_size': 16,
            'epochs': 50,
            'confidence_threshold': 0.8,
            'temperature': 1.0,
            'labeled_ratio': 0.3,  # Fraction of data that is labeled
            'validation_split': 0.2,
            'test_split': 0.15,
            'augmentation_prob': 0.8,
            'checkpoint_dir': 'checkpoints/mpl',
            'log_dir': 'logs/mpl',
            'results_dir': 'results/mpl',
            'save_frequency': 10
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
    
    def build_models(self):
        """Build teacher and student models"""
        print("Building teacher and student models...")
        
        # Teacher model (typically pre-trained or larger)
        self.teacher_model = EfficientNetClassifier(
            num_classes=self.config['num_classes'],
            input_shape=self.config['input_shape'],
            efficientnet_version=self.config['efficientnet_version'],
            freeze_base=False,  # Teacher can be more flexible
            dropout_rate=0.1
        )
        
        # Student model (typically smaller or same size)
        self.student_model = EfficientNetClassifier(
            num_classes=self.config['num_classes'],
            input_shape=self.config['input_shape'],
            efficientnet_version=self.config['efficientnet_version'],
            freeze_base=True,  # Student starts with frozen base
            dropout_rate=0.2
        )
        
        # Compile models
        self.teacher_model.compile_model(
            optimizer='adam',
            learning_rate=self.config['teacher_learning_rate']
        )
        
        self.student_model.compile_model(
            optimizer='adam',
            learning_rate=self.config['student_learning_rate']
        )
        
        # Create MPL framework
        self.mpl_framework = MetaPseudoLabels(
            teacher_model=self.teacher_model.model,
            student_model=self.student_model.model,
            num_classes=self.config['num_classes'],
            confidence_threshold=self.config['confidence_threshold'],
            temperature=self.config['temperature']
        )
        
        print("Models built successfully!")
        print(f"Teacher parameters: {self.teacher_model.model.count_params():,}")
        print(f"Student parameters: {self.student_model.model.count_params():,}")
        
        return self.mpl_framework
    
    def prepare_data(self, images, labels, preprocess=True, augment=True):
        """
        Prepare data for MPL training
        
        Args:
            images: Array of input images
            labels: Array of corresponding labels
            preprocess: Whether to apply preprocessing
            augment: Whether to apply data augmentation
            
        Returns:
            Prepared datasets (labeled, unlabeled, validation, test)
        """
        print("Preparing data for MPL training...")
        
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
        
        # Split data into train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=self.config['test_split'],
            random_state=42,
            stratify=labels
        )
        
        X_train_all, X_val, y_train_all, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config['validation_split'] / (1 - self.config['test_split']),
            random_state=42,
            stratify=y_temp
        )
        
        # Create labeled/unlabeled split from training data
        labeled_indices, unlabeled_indices = train_test_split(
            np.arange(len(X_train_all)),
            test_size=1 - self.config['labeled_ratio'],
            random_state=42,
            stratify=y_train_all
        )
        
        # Labeled data
        X_labeled = X_train_all[labeled_indices]
        y_labeled = y_train_all[labeled_indices]
        
        # Unlabeled data (we have labels but won't use them during training)
        X_unlabeled = X_train_all[unlabeled_indices]
        y_unlabeled = y_train_all[unlabeled_indices]  # For evaluation only
        
        print(f"Labeled samples: {len(X_labeled)}")
        print(f"Unlabeled samples: {len(X_unlabeled)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Data augmentation
        if augment:
            print("Applying data augmentation...")
            
            # Augment labeled data
            augmented_labeled = self.augmentation.augment_dataset(
                X_labeled, y_labeled, augmentation_factor=2
            )
            X_labeled = augmented_labeled['images']
            y_labeled = augmented_labeled['labels']
            
            # Augment unlabeled data
            augmented_unlabeled = self.augmentation.augment_dataset(
                X_unlabeled, y_unlabeled, augmentation_factor=3
            )
            X_unlabeled = augmented_unlabeled['images']
            
            print(f"Augmented labeled samples: {len(X_labeled)}")
            print(f"Augmented unlabeled samples: {len(X_unlabeled)}")
        
        # Store data for later use
        self.data = {
            'labeled': (X_labeled, y_labeled),
            'unlabeled': X_unlabeled,
            'validation': (X_val, y_val),
            'test': (X_test, y_test),
            'unlabeled_true_labels': y_unlabeled  # For evaluation
        }
        
        return self.data
    
    def pretrain_teacher(self, epochs=10):
        """
        Pre-train the teacher model on labeled data
        
        Args:
            epochs: Number of pre-training epochs
            
        Returns:
            Pre-training history
        """
        print("Pre-training teacher model...")
        
        X_labeled, y_labeled = self.data['labeled']
        X_val, y_val = self.data['validation']
        
        # Convert labels to categorical
        y_labeled_cat = tf.keras.utils.to_categorical(y_labeled, self.config['num_classes'])
        y_val_cat = tf.keras.utils.to_categorical(y_val, self.config['num_classes'])
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_labeled, y_labeled_cat))
        train_dataset = train_dataset.batch(self.config['batch_size']).shuffle(1000)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))
        val_dataset = val_dataset.batch(self.config['batch_size'])
        
        # Pre-train teacher
        history = self.teacher_model.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=1,
            callbacks=[
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    verbose=1
                )
            ]
        )
        
        print("Teacher pre-training completed!")
        
        return history
    
    def train_mpl(self, use_teacher_pretraining=True, pretraining_epochs=10):
        """
        Train the Meta Pseudo Labels framework
        
        Args:
            use_teacher_pretraining: Whether to pre-train teacher on labeled data
            pretraining_epochs: Number of pre-training epochs
            
        Returns:
            Training history
        """
        if self.mpl_framework is None:
            raise ValueError("MPL framework not built. Call build_models() first.")
        
        print("Starting Meta Pseudo Labels training...")
        
        # Pre-train teacher if requested
        pretraining_history = None
        if use_teacher_pretraining:
            pretraining_history = self.pretrain_teacher(epochs=pretraining_epochs)
        
        # Prepare data for MPL training
        X_labeled, y_labeled = self.data['labeled']
        X_unlabeled = self.data['unlabeled']
        X_val, y_val = self.data['validation']
        
        # Train MPL framework
        mpl_history = self.mpl_framework.train(
            labeled_data=X_labeled,
            labeled_labels=y_labeled,
            unlabeled_data=X_unlabeled,
            validation_data=X_val,
            validation_labels=y_val,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size']
        )
        
        # Save models periodically during training
        self._save_checkpoint_during_training(mpl_history)
        
        print("MPL training completed!")
        
        return {
            'pretraining_history': pretraining_history,
            'mpl_history': mpl_history
        }
    
    def _save_checkpoint_during_training(self, history):
        """Save model checkpoints during training"""
        # Save final models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        teacher_path = os.path.join(
            self.config['checkpoint_dir'],
            f'teacher_model_{timestamp}.h5'
        )
        student_path = os.path.join(
            self.config['checkpoint_dir'],
            f'student_model_{timestamp}.h5'
        )
        
        self.mpl_framework.save_models(teacher_path, student_path)
        
        # Save training history
        history_path = os.path.join(
            self.config['results_dir'],
            f'mpl_history_{timestamp}.json'
        )
        
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_history = {}
            for key, value in history.items():
                if isinstance(value, np.ndarray):
                    json_history[key] = value.tolist()
                elif isinstance(value, list):
                    json_history[key] = value
                else:
                    json_history[key] = str(value)
            
            json.dump(json_history, f, indent=2)
        
        print(f"Checkpoint saved: {teacher_path}, {student_path}")
    
    def evaluate_mpl(self, save_results=True):
        """
        Evaluate the MPL framework
        
        Args:
            save_results: Whether to save evaluation results
            
        Returns:
            Evaluation metrics for both teacher and student
        """
        if self.mpl_framework is None:
            raise ValueError("MPL framework not trained. Train the model first.")
        
        print("Evaluating MPL framework...")
        
        X_test, y_test = self.data['test']
        
        # Evaluate teacher model
        teacher_predictions = self.teacher_model.model.predict(X_test)
        teacher_pred_classes = np.argmax(teacher_predictions, axis=1)
        
        teacher_metrics = self.metrics_calculator.calculate_metrics(
            y_test, teacher_pred_classes, teacher_predictions
        )
        
        # Evaluate student model
        student_predictions = self.student_model.model.predict(X_test)
        student_pred_classes = np.argmax(student_predictions, axis=1)
        
        student_metrics = self.metrics_calculator.calculate_metrics(
            y_test, student_pred_classes, student_predictions
        )
        
        # Print results
        print("\nMPL Evaluation Results:")
        print("=" * 50)
        print("TEACHER MODEL:")
        print(f"Accuracy: {teacher_metrics['accuracy']:.4f}")
        print(f"Precision (macro): {teacher_metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {teacher_metrics['recall_macro']:.4f}")
        print(f"F1-score (macro): {teacher_metrics['f1_macro']:.4f}")
        
        print("\nSTUDENT MODEL:")
        print(f"Accuracy: {student_metrics['accuracy']:.4f}")
        print(f"Precision (macro): {student_metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {student_metrics['recall_macro']:.4f}")
        print(f"F1-score (macro): {student_metrics['f1_macro']:.4f}")
        
        # Compare with supervised baseline (teacher trained only on labeled data)
        print("\nIMPROVEMENT ANALYSIS:")
        accuracy_improvement = student_metrics['accuracy'] - teacher_metrics['accuracy']
        print(f"Student vs Teacher Accuracy: {accuracy_improvement:+.4f}")
        
        if save_results:
            self._save_mpl_evaluation_results(teacher_metrics, student_metrics, 
                                            y_test, teacher_pred_classes, student_pred_classes,
                                            teacher_predictions, student_predictions)
        
        return {
            'teacher_metrics': teacher_metrics,
            'student_metrics': student_metrics,
            'improvement': {
                'accuracy': accuracy_improvement,
                'precision': student_metrics['precision_macro'] - teacher_metrics['precision_macro'],
                'recall': student_metrics['recall_macro'] - teacher_metrics['recall_macro'],
                'f1_score': student_metrics['f1_macro'] - teacher_metrics['f1_macro']
            }
        }
    
    def _save_mpl_evaluation_results(self, teacher_metrics, student_metrics, 
                                   y_true, teacher_pred, student_pred,
                                   teacher_prob, student_prob):
        """Save MPL evaluation results and visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Confusion matrices
        teacher_cm_fig = self.visualizer.plot_confusion_matrix_interactive(
            y_true, teacher_pred, self.config['class_names'],
            title="Teacher Model Confusion Matrix"
        )
        teacher_cm_fig.write_html(
            os.path.join(self.config['results_dir'], f'teacher_confusion_matrix_{timestamp}.html')
        )
        
        student_cm_fig = self.visualizer.plot_confusion_matrix_interactive(
            y_true, student_pred, self.config['class_names'],
            title="Student Model Confusion Matrix"
        )
        student_cm_fig.write_html(
            os.path.join(self.config['results_dir'], f'student_confusion_matrix_{timestamp}.html')
        )
        
        # ROC curves comparison
        comparison_fig = self._create_model_comparison_plot(
            y_true, teacher_prob, student_prob
        )
        comparison_fig.write_html(
            os.path.join(self.config['results_dir'], f'model_comparison_{timestamp}.html')
        )
        
        # Save metrics
        combined_metrics = {
            'teacher': teacher_metrics,
            'student': student_metrics,
            'improvement': {
                'accuracy': student_metrics['accuracy'] - teacher_metrics['accuracy'],
                'precision': student_metrics['precision_macro'] - teacher_metrics['precision_macro'],
                'recall': student_metrics['recall_macro'] - teacher_metrics['recall_macro'],
                'f1_score': student_metrics['f1_macro'] - teacher_metrics['f1_macro']
            }
        }
        
        # Convert numpy types for JSON serialization
        json_metrics = self._convert_metrics_for_json(combined_metrics)
        
        with open(
            os.path.join(self.config['results_dir'], f'mpl_evaluation_{timestamp}.json'),
            'w'
        ) as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"MPL evaluation results saved to {self.config['results_dir']}")
    
    def _convert_metrics_for_json(self, metrics):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(metrics, dict):
            return {k: self._convert_metrics_for_json(v) for k, v in metrics.items() 
                   if k != 'confusion_matrix' and k != 'classification_report'}
        elif isinstance(metrics, np.ndarray):
            return metrics.tolist()
        elif isinstance(metrics, (np.int64, np.float64)):
            return metrics.item()
        else:
            return metrics
    
    def _create_model_comparison_plot(self, y_true, teacher_prob, student_prob):
        """Create comparison plot for teacher and student models"""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        import plotly.graph_objects as go
        
        # Convert to binary format for ROC calculation
        y_true_bin = label_binarize(y_true, classes=range(self.config['num_classes']))
        
        fig = go.Figure()
        
        # Plot ROC curves for each class and model
        for i, class_name in enumerate(self.config['class_names']):
            # Teacher ROC
            fpr_teacher, tpr_teacher, _ = roc_curve(y_true_bin[:, i], teacher_prob[:, i])
            auc_teacher = auc(fpr_teacher, tpr_teacher)
            
            fig.add_trace(go.Scatter(
                x=fpr_teacher, y=tpr_teacher,
                mode='lines',
                name=f'Teacher {class_name} (AUC = {auc_teacher:.2f})',
                line=dict(dash='solid')
            ))
            
            # Student ROC
            fpr_student, tpr_student, _ = roc_curve(y_true_bin[:, i], student_prob[:, i])
            auc_student = auc(fpr_student, tpr_student)
            
            fig.add_trace(go.Scatter(
                x=fpr_student, y=tpr_student,
                mode='lines',
                name=f'Student {class_name} (AUC = {auc_student:.2f})',
                line=dict(dash='dash')
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dot', color='gray')
        ))
        
        fig.update_layout(
            title='Teacher vs Student Model ROC Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template='plotly_white'
        )
        
        return fig
    
    def analyze_pseudo_label_quality(self):
        """
        Analyze the quality of generated pseudo labels
        
        Returns:
            Pseudo label analysis results
        """
        if self.mpl_framework is None:
            raise ValueError("MPL framework not trained.")
        
        print("Analyzing pseudo label quality...")
        
        X_unlabeled = self.data['unlabeled']
        y_unlabeled_true = self.data['unlabeled_true_labels']
        
        # Generate pseudo labels
        pseudo_labels, confidence_mask = self.mpl_framework.generate_pseudo_labels(X_unlabeled)
        pseudo_pred_classes = np.argmax(pseudo_labels, axis=1)
        
        # Calculate accuracy of pseudo labels
        high_confidence_indices = np.where(confidence_mask)[0]
        
        if len(high_confidence_indices) > 0:
            pseudo_accuracy = np.mean(
                pseudo_pred_classes[high_confidence_indices] == y_unlabeled_true[high_confidence_indices]
            )
            
            analysis = {
                'total_unlabeled_samples': len(X_unlabeled),
                'high_confidence_samples': len(high_confidence_indices),
                'high_confidence_ratio': len(high_confidence_indices) / len(X_unlabeled),
                'pseudo_label_accuracy': pseudo_accuracy,
                'confidence_threshold': self.config['confidence_threshold']
            }
            
            print(f"Pseudo Label Analysis:")
            print(f"High confidence samples: {analysis['high_confidence_samples']}/{analysis['total_unlabeled_samples']} ({analysis['high_confidence_ratio']:.2%})")
            print(f"Pseudo label accuracy: {analysis['pseudo_label_accuracy']:.4f}")
            
        else:
            analysis = {
                'total_unlabeled_samples': len(X_unlabeled),
                'high_confidence_samples': 0,
                'high_confidence_ratio': 0.0,
                'pseudo_label_accuracy': 0.0,
                'confidence_threshold': self.config['confidence_threshold']
            }
            print("No high confidence pseudo labels generated.")
        
        return analysis
    
    def load_mpl_models(self, teacher_path, student_path):
        """
        Load trained MPL models
        
        Args:
            teacher_path: Path to teacher model
            student_path: Path to student model
        """
        if self.mpl_framework is None:
            self.build_models()
        
        self.mpl_framework.load_models(teacher_path, student_path)
        print(f"MPL models loaded from {teacher_path} and {student_path}")
    
    def predict_with_mpl(self, image, use_student=True, preprocess=True):
        """
        Predict using the trained MPL models
        
        Args:
            image: Input image
            use_student: Whether to use student model (default) or teacher
            preprocess: Whether to preprocess the image
            
        Returns:
            Prediction results
        """
        if self.mpl_framework is None:
            raise ValueError("MPL framework not trained.")
        
        # Select model
        model = self.student_model if use_student else self.teacher_model
        
        # Get prediction
        prediction = model.predict_single_image(image, preprocess=preprocess)
        
        return {
            'model_type': 'Student' if use_student else 'Teacher',
            **prediction
        }
    
    def get_training_summary(self):
        """
        Get a summary of the MPL training configuration
        
        Returns:
            Training summary dictionary
        """
        return {
            'framework': 'Meta Pseudo Labels',
            'teacher_architecture': f"EfficientNet{self.config['efficientnet_version']}",
            'student_architecture': f"EfficientNet{self.config['efficientnet_version']}",
            'num_classes': self.config['num_classes'],
            'input_shape': self.config['input_shape'],
            'confidence_threshold': self.config['confidence_threshold'],
            'temperature': self.config['temperature'],
            'labeled_ratio': self.config['labeled_ratio'],
            'batch_size': self.config['batch_size'],
            'epochs': self.config['epochs'],
            'teacher_learning_rate': self.config['teacher_learning_rate'],
            'student_learning_rate': self.config['student_learning_rate'],
            'class_names': self.config['class_names']
        }

