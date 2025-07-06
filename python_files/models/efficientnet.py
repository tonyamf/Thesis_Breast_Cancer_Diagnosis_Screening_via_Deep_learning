import tensorflow as tf
from tensorflow.keras import layers, Model, applications
import numpy as np

class EfficientNetClassifier:
    """
    EfficientNet implementation for breast cancer classification
    Based on EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
    """
    
    def __init__(self, num_classes=3, input_shape=(224, 224, 3), 
                 efficientnet_version='B0', freeze_base=True, dropout_rate=0.2):
        """
        Initialize EfficientNet classifier
        
        Args:
            num_classes: Number of output classes (3 for Normal, Benign, Malignant)
            input_shape: Shape of input images
            efficientnet_version: Version of EfficientNet to use (B0-B7)
            freeze_base: Whether to freeze base model weights
            dropout_rate: Dropout rate for regularization
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.efficientnet_version = efficientnet_version
        self.freeze_base = freeze_base
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
    
    def _get_base_model(self):
        """
        Get the base EfficientNet model
        
        Returns:
            Base EfficientNet model
        """
        # Map version strings to actual models
        model_map = {
            'B0': applications.EfficientNetB0,
            'B1': applications.EfficientNetB1,
            'B2': applications.EfficientNetB2,
            'B3': applications.EfficientNetB3,
            'B4': applications.EfficientNetB4,
            'B5': applications.EfficientNetB5,
            'B6': applications.EfficientNetB6,
            'B7': applications.EfficientNetB7,
        }
        
        if self.efficientnet_version not in model_map:
            raise ValueError(f"Unsupported EfficientNet version: {self.efficientnet_version}")
        
        # Load pre-trained model
        base_model = model_map[self.efficientnet_version](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model if specified
        if self.freeze_base:
            base_model.trainable = False
        
        return base_model
    
    def _build_model(self):
        """
        Build the complete EfficientNet classification model
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Preprocessing for EfficientNet
        # EfficientNet expects inputs in range [0, 255]
        x = layers.Rescaling(255.0)(inputs)
        x = applications.efficientnet.preprocess_input(x)
        
        # Base EfficientNet model
        base_model = self._get_base_model()
        x = base_model(x, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers for classification
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        if self.num_classes == 2:
            # Binary classification
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            # Multi-class classification
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=f'EfficientNet{self.efficientnet_version}_Classifier')
        
        return model
    
    def compile_model(self, optimizer='adam', learning_rate=1e-4):
        """
        Compile the model with appropriate loss and metrics
        
        Args:
            optimizer: Optimizer for training
            learning_rate: Learning rate for optimizer
        """
        # Set up optimizer
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        # Set up loss and metrics based on number of classes
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', 'top_k_categorical_accuracy']
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
    
    def fine_tune(self, train_generator, validation_generator, 
                  fine_tune_epochs=10, initial_epochs=10):
        """
        Fine-tune the pre-trained model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            fine_tune_epochs: Number of fine-tuning epochs
            initial_epochs: Number of initial training epochs
            
        Returns:
            Training history
        """
        # Step 1: Train with frozen base
        print("Step 1: Training with frozen base model...")
        
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=initial_epochs,
            verbose=1
        )
        
        # Step 2: Unfreeze and fine-tune
        print("Step 2: Fine-tuning with unfrozen base model...")
        
        # Unfreeze the base model
        base_model = self.model.layers[3]  # Assuming base model is the 4th layer
        base_model.trainable = True
        
        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=self.model.loss,
            metrics=self.model.metrics
        )
        
        # Continue training
        fine_tune_history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=initial_epochs + fine_tune_epochs,
            initial_epoch=initial_epochs,
            verbose=1
        )
        
        # Combine histories
        for key in history.history:
            history.history[key].extend(fine_tune_history.history[key])
        
        return history
    
    def train(self, train_generator, validation_generator, epochs=50, 
              callbacks=None, verbose=1):
        """
        Train the EfficientNet model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            callbacks: List of Keras callbacks
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Default callbacks
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    'best_efficientnet_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
        
        # Train the model
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, image):
        """
        Predict class probabilities for input image
        
        Args:
            image: Input image array
            
        Returns:
            Predicted class probabilities
        """
        # Ensure image has correct shape
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Ensure correct input shape
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        # Resize to model input size
        if image.shape[1:3] != self.input_shape[:2]:
            image = tf.image.resize(image, self.input_shape[:2])
        
        # Normalize to [0, 1] range
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Predict
        predictions = self.model.predict(image)
        
        return predictions
    
    def predict_class(self, image):
        """
        Predict class label for input image
        
        Args:
            image: Input image array
            
        Returns:
            Predicted class index and confidence
        """
        predictions = self.predict(image)
        
        if self.num_classes == 2:
            # Binary classification
            confidence = predictions[0, 0]
            predicted_class = int(confidence > 0.5)
            confidence = confidence if predicted_class == 1 else 1 - confidence
        else:
            # Multi-class classification
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0, predicted_class]
        
        return predicted_class, confidence
    
    def evaluate_model(self, test_generator):
        """
        Evaluate model performance on test data
        
        Args:
            test_generator: Test data generator
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Get predictions and true labels
        predictions = self.model.predict(test_generator)
        true_labels = test_generator.classes
        
        if self.num_classes == 2:
            # Binary classification metrics
            predicted_labels = (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-class classification metrics
            predicted_labels = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        
        # Classification report
        class_names = ['Normal', 'Benign', 'Malignant'][:self.num_classes]
        report = classification_report(true_labels, predicted_labels, 
                                     target_names=class_names, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report
        }
    
    def load_weights(self, filepath):
        """
        Load model weights from file
        
        Args:
            filepath: Path to weights file
        """
        self.model.load_weights(filepath)
    
    def save_weights(self, filepath):
        """
        Save model weights to file
        
        Args:
            filepath: Path to save weights
        """
        self.model.save_weights(filepath)
    
    def get_model_summary(self):
        """
        Get model architecture summary
        
        Returns:
            Model summary string
        """
        return self.model.summary()
    
    def visualize_feature_maps(self, image, layer_names=None):
        """
        Visualize feature maps from intermediate layers
        
        Args:
            image: Input image
            layer_names: List of layer names to visualize
            
        Returns:
            Dictionary of feature maps
        """
        if layer_names is None:
            # Select some representative layers
            layer_names = ['block2a_expand_conv', 'block4a_expand_conv', 
                          'block6a_expand_conv', 'top_conv']
        
        # Create model for feature extraction
        feature_extractor = Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(name).output for name in layer_names]
        )
        
        # Get feature maps
        feature_maps = feature_extractor.predict(np.expand_dims(image, axis=0))
        
        return {name: feature_map for name, feature_map in zip(layer_names, feature_maps)}
