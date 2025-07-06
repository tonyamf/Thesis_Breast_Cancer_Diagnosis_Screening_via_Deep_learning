import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class ClassificationMetrics:
    """
    Comprehensive metrics for classification tasks
    """
    
    def __init__(self, num_classes=3, class_names=None):
        """
        Initialize classification metrics
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # AUC-ROC (if probabilities provided)
        if y_prob is not None:
            if self.num_classes == 2:
                # Binary classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multi-class classification
                try:
                    metrics['auc_roc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                    metrics['auc_roc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo')
                except ValueError:
                    # Handle case where not all classes are present
                    metrics['auc_roc_ovr'] = None
                    metrics['auc_roc_ovo'] = None
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0
        )
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, title='Confusion Matrix'):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the matrix
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_curves(self, y_true, y_prob, title='ROC Curves'):
        """
        Plot ROC curves for multi-class classification
        
        Args:
            y_true: True labels (one-hot encoded)
            y_prob: Predicted probabilities
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 8))
        
        if self.num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            auc = roc_auc_score(y_true, y_prob[:, 1])
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
        else:
            # Multi-class classification
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {auc:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_precision_recall_curves(self, y_true, y_prob, title='Precision-Recall Curves'):
        """
        Plot precision-recall curves
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 8))
        
        if self.num_classes == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            plt.plot(recall, precision, label='Precision-Recall Curve')
        else:
            # Multi-class classification
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            for i in range(self.num_classes):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                plt.plot(recall, precision, label=f'{self.class_names[i]}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()

class SegmentationMetrics:
    """
    Comprehensive metrics for segmentation tasks
    """
    
    def __init__(self):
        """Initialize segmentation metrics"""
        pass
    
    def dice_coefficient(self, y_true, y_pred, smooth=1e-6):
        """
        Calculate Dice coefficient
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            smooth: Smoothing factor
            
        Returns:
            Dice coefficient
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice
    
    def iou_score(self, y_true, y_pred, smooth=1e-6):
        """
        Calculate Intersection over Union (IoU) score
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            smooth: Smoothing factor
            
        Returns:
            IoU score
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou
    
    def hausdorff_distance(self, y_true, y_pred):
        """
        Calculate Hausdorff distance between binary masks
        
        Args:
            y_true: Ground truth mask
            y_pred: Predicted mask
            
        Returns:
            Hausdorff distance
        """
        from scipy.spatial.distance import directed_hausdorff
        
        # Get boundary points
        true_points = np.column_stack(np.where(y_true > 0))
        pred_points = np.column_stack(np.where(y_pred > 0))
        
        if len(true_points) == 0 or len(pred_points) == 0:
            return float('inf')
        
        # Calculate directed Hausdorff distances
        d1 = directed_hausdorff(true_points, pred_points)[0]
        d2 = directed_hausdorff(pred_points, true_points)[0]
        
        # Return maximum
        return max(d1, d2)
    
    def sensitivity(self, y_true, y_pred):
        """
        Calculate sensitivity (recall)
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            
        Returns:
            Sensitivity score
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
        true_positives = np.sum(y_true_f * y_pred_f)
        false_negatives = np.sum(y_true_f * (1 - y_pred_f))
        
        sensitivity = true_positives / (true_positives + false_negatives + 1e-6)
        return sensitivity
    
    def specificity(self, y_true, y_pred):
        """
        Calculate specificity
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            
        Returns:
            Specificity score
        """
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
        true_negatives = np.sum((1 - y_true_f) * (1 - y_pred_f))
        false_positives = np.sum((1 - y_true_f) * y_pred_f)
        
        specificity = true_negatives / (true_negatives + false_positives + 1e-6)
        return specificity
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive segmentation metrics
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            
        Returns:
            Dictionary of metrics
        """
        # Ensure binary masks
        y_true_binary = (y_true > 0.5).astype(np.float32)
        y_pred_binary = (y_pred > 0.5).astype(np.float32)
        
        metrics = {}
        
        # Calculate metrics
        metrics['dice_coefficient'] = self.dice_coefficient(y_true_binary, y_pred_binary)
        metrics['iou_score'] = self.iou_score(y_true_binary, y_pred_binary)
        metrics['sensitivity'] = self.sensitivity(y_true_binary, y_pred_binary)
        metrics['specificity'] = self.specificity(y_true_binary, y_pred_binary)
        
        # Pixel accuracy
        metrics['pixel_accuracy'] = np.mean(y_true_binary == y_pred_binary)
        
        # Hausdorff distance (computationally expensive)
        try:
            metrics['hausdorff_distance'] = self.hausdorff_distance(y_true_binary, y_pred_binary)
        except:
            metrics['hausdorff_distance'] = None
        
        return metrics
    
    def plot_segmentation_results(self, image, y_true, y_pred, title='Segmentation Results'):
        """
        Plot segmentation results
        
        Args:
            image: Original image
            y_true: Ground truth mask
            y_pred: Predicted mask
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(y_true, cmap='hot', alpha=0.7)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(y_pred, cmap='hot', alpha=0.7)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Overlay
        axes[3].imshow(image, cmap='gray')
        axes[3].imshow(y_pred, cmap='hot', alpha=0.5)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig

class TensorFlowMetrics:
    """
    TensorFlow custom metrics for training
    """
    
    @staticmethod
    def dice_coefficient_tf(y_true, y_pred, smooth=1e-6):
        """
        TensorFlow implementation of Dice coefficient
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            smooth: Smoothing factor
            
        Returns:
            Dice coefficient tensor
        """
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice
    
    @staticmethod
    def iou_score_tf(y_true, y_pred, smooth=1e-6):
        """
        TensorFlow implementation of IoU score
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            smooth: Smoothing factor
            
        Returns:
            IoU score tensor
        """
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou
    
    @staticmethod
    def precision_tf(y_true, y_pred):
        """
        TensorFlow implementation of precision
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            
        Returns:
            Precision tensor
        """
        true_positives = tf.reduce_sum(tf.cast(y_true * tf.round(y_pred), tf.float32))
        predicted_positives = tf.reduce_sum(tf.cast(tf.round(y_pred), tf.float32))
        
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision
    
    @staticmethod
    def recall_tf(y_true, y_pred):
        """
        TensorFlow implementation of recall
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            
        Returns:
            Recall tensor
        """
        true_positives = tf.reduce_sum(tf.cast(y_true * tf.round(y_pred), tf.float32))
        possible_positives = tf.reduce_sum(tf.cast(y_true, tf.float32))
        
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall
    
    @staticmethod
    def f1_score_tf(y_true, y_pred):
        """
        TensorFlow implementation of F1 score
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            
        Returns:
            F1 score tensor
        """
        precision = TensorFlowMetrics.precision_tf(y_true, y_pred)
        recall = TensorFlowMetrics.recall_tf(y_true, y_pred)
        
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return f1
