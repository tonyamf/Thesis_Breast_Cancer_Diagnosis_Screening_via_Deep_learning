import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

class ResultVisualizer:
    """
    Comprehensive visualization tools for breast cancer diagnosis results
    """
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize the visualizer
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.class_names = ['Normal', 'Benign', 'Malignant']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_training_history(self, history, metrics=['loss', 'accuracy'], title="Training History"):
        """
        Plot training history with multiple metrics
        
        Args:
            history: Training history dictionary
            metrics: List of metrics to plot
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=[f'{metric.title()} Over Epochs' for metric in metrics],
            vertical_spacing=0.08
        )
        
        epochs = list(range(1, len(history[metrics[0]]) + 1))
        
        for i, metric in enumerate(metrics):
            # Training metric
            if metric in history:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history[metric],
                        mode='lines+markers',
                        name=f'Training {metric.title()}',
                        line=dict(color=self.colors[0]),
                        marker=dict(size=4)
                    ),
                    row=i+1, col=1
                )
            
            # Validation metric
            val_metric = f'val_{metric}'
            if val_metric in history:
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history[val_metric],
                        mode='lines+markers',
                        name=f'Validation {metric.title()}',
                        line=dict(color=self.colors[1]),
                        marker=dict(size=4)
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title=title,
            height=300 * len(metrics),
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Epoch")
        
        return fig
    
    def plot_confusion_matrix_interactive(self, y_true, y_pred, class_names=None, 
                                        normalize=False, title="Confusion Matrix"):
        """
        Create interactive confusion matrix with Plotly
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            normalize: Whether to normalize the matrix
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if class_names is None:
            class_names = self.class_names
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            text_template = '%{z:.2f}'
        else:
            text_template = '%{z:d}'
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate=text_template,
            textfont={"size": 16},
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=600,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def plot_roc_curves_interactive(self, y_true, y_prob, class_names=None, 
                                  title="ROC Curves"):
        """
        Create interactive ROC curves
        
        Args:
            y_true: True labels (one-hot encoded or integer)
            y_prob: Predicted probabilities
            class_names: List of class names
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if class_names is None:
            class_names = self.class_names
        
        fig = go.Figure()
        
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            from sklearn.preprocessing import label_binarize
            n_classes = len(np.unique(y_true))
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
        else:
            y_true_bin = y_true
            n_classes = y_true.shape[1]
        
        # Plot ROC curve for each class
        for i in range(min(n_classes, len(class_names))):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{class_names[i]} (AUC = {roc_auc:.2f})',
                line=dict(width=2)
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray', width=1)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700,
            height=500,
            template='plotly_white',
            legend=dict(x=0.6, y=0.05)
        )
        
        return fig
    
    def plot_segmentation_comparison(self, original_image, ground_truth, prediction, 
                                   title="Segmentation Results"):
        """
        Create side-by-side comparison of segmentation results
        
        Args:
            original_image: Original mammography image
            ground_truth: Ground truth segmentation mask
            prediction: Predicted segmentation mask
            title: Plot title
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=['Original', 'Ground Truth', 'Prediction', 'Overlay'],
            horizontal_spacing=0.05
        )
        
        # Original image
        fig.add_trace(
            go.Heatmap(z=original_image, colorscale='gray', showscale=False),
            row=1, col=1
        )
        
        # Ground truth
        fig.add_trace(
            go.Heatmap(z=ground_truth, colorscale='Reds', showscale=False),
            row=1, col=2
        )
        
        # Prediction
        fig.add_trace(
            go.Heatmap(z=prediction, colorscale='Reds', showscale=False),
            row=1, col=3
        )
        
        # Overlay (original + prediction)
        overlay = original_image.copy()
        overlay[prediction > 0.5] = overlay[prediction > 0.5] * 0.7 + prediction[prediction > 0.5] * 0.3
        fig.add_trace(
            go.Heatmap(z=overlay, colorscale='gray', showscale=False),
            row=1, col=4
        )
        
        fig.update_layout(
            title=title,
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        
        # Remove axis labels and ticks
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return fig
    
    def plot_model_performance_comparison(self, model_results, metrics=['accuracy', 'precision', 'recall', 'f1_score']):
        """
        Compare performance of different models
        
        Args:
            model_results: Dictionary with model names as keys and metrics as values
            metrics: List of metrics to compare
            
        Returns:
            Plotly figure
        """
        models = list(model_results.keys())
        
        fig = go.Figure()
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [model_results[model].get(metric, 0) for model in models]
            
            fig.add_trace(go.Bar(
                x=[f"{model}_{metric}" for model in models],
                y=values,
                name=metric.title(),
                offsetgroup=i,
                marker_color=self.colors[i % len(self.colors)]
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_class_distribution(self, labels, class_names=None, title="Class Distribution"):
        """
        Plot distribution of classes in the dataset
        
        Args:
            labels: Array of class labels
            class_names: List of class names
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if class_names is None:
            class_names = self.class_names
        
        unique, counts = np.unique(labels, return_counts=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=[class_names[i] if i < len(class_names) else f'Class {i}' for i in unique],
                y=counts,
                marker_color=self.colors[:len(unique)],
                text=counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Classes",
            yaxis_title="Number of Samples",
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_feature_importance(self, feature_names, importance_scores, title="Feature Importance"):
        """
        Plot feature importance scores
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_scores[:20],  # Top 20 features
                y=sorted_features[:20],
                orientation='h',
                marker_color=self.colors[0]
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def plot_learning_curves(self, train_sizes, train_scores, val_scores, 
                           title="Learning Curves"):
        """
        Plot learning curves showing training and validation scores vs training set size
        
        Args:
            train_sizes: Array of training set sizes
            train_scores: Training scores for each size
            val_scores: Validation scores for each size
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(train_scores, axis=1),
            mode='lines+markers',
            name='Training Score',
            line=dict(color=self.colors[0]),
            error_y=dict(
                type='data',
                array=np.std(train_scores, axis=1),
                visible=True
            )
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(val_scores, axis=1),
            mode='lines+markers',
            name='Validation Score',
            line=dict(color=self.colors[1]),
            error_y=dict(
                type='data',
                array=np.std(val_scores, axis=1),
                visible=True
            )
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Training Set Size",
            yaxis_title="Score",
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_loss_landscape(self, loss_surface, title="Loss Landscape"):
        """
        Plot 2D loss landscape visualization
        
        Args:
            loss_surface: 2D array representing loss values
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Contour(
            z=loss_surface,
            colorscale='Viridis',
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white')
            )
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Parameter 1",
            yaxis_title="Parameter 2",
            template='plotly_white',
            height=500,
            width=500
        )
        
        return fig
    
    def plot_attention_maps(self, image, attention_weights, title="Attention Visualization"):
        """
        Visualize attention maps overlaid on the original image
        
        Args:
            image: Original image
            attention_weights: Attention weight maps
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Original Image', 'Attention Map'],
            horizontal_spacing=0.1
        )
        
        # Original image
        fig.add_trace(
            go.Heatmap(z=image, colorscale='gray', showscale=False),
            row=1, col=1
        )
        
        # Attention map overlay
        fig.add_trace(
            go.Heatmap(z=attention_weights, colorscale='hot', showscale=True),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            height=400,
            template='plotly_white'
        )
        
        # Remove axis labels
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return fig
    
    def create_model_architecture_diagram(self, layer_info, title="Model Architecture"):
        """
        Create a visual representation of model architecture
        
        Args:
            layer_info: List of dictionaries with layer information
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        y_pos = 0
        layer_height = 1
        layer_spacing = 0.5
        
        for i, layer in enumerate(layer_info):
            # Draw layer box
            fig.add_shape(
                type="rect",
                x0=0, y0=y_pos,
                x1=layer.get('width', 2), y1=y_pos + layer_height,
                fillcolor=self.colors[i % len(self.colors)],
                opacity=0.7,
                line=dict(color="black", width=1)
            )
            
            # Add layer label
            fig.add_annotation(
                x=layer.get('width', 2) / 2,
                y=y_pos + layer_height / 2,
                text=f"{layer['name']}<br>{layer.get('shape', '')}",
                showarrow=False,
                font=dict(size=10, color="white")
            )
            
            y_pos += layer_height + layer_spacing
        
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 3]),
            yaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, y_pos]),
            template='plotly_white',
            height=max(400, y_pos * 50),
            width=400
        )
        
        return fig
    
    def save_figure(self, fig, filename, format='png', width=800, height=600):
        """
        Save figure to file
        
        Args:
            fig: Plotly or matplotlib figure
            filename: Output filename
            format: File format ('png', 'pdf', 'svg', 'html')
            width: Figure width
            height: Figure height
        """
        if hasattr(fig, 'write_image'):
            # Plotly figure
            if format == 'html':
                fig.write_html(filename)
            else:
                fig.write_image(filename, format=format, width=width, height=height)
        else:
            # Matplotlib figure
            fig.savefig(filename, format=format, dpi=300, bbox_inches='tight')

