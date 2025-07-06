import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class UNet:
    """
    U-Net implementation for medical image segmentation
    Based on the original U-Net paper: https://arxiv.org/abs/1505.04597
    """
    
    def __init__(self, input_shape=(512, 512, 1), num_classes=1, filters=64):
        """
        Initialize U-Net model
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes (1 for binary segmentation)
            filters: Number of filters in the first layer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.model = self._build_model()
    
    def _conv_block(self, inputs, filters, kernel_size=3, activation='relu', 
                   batch_norm=True, dropout_rate=0.0):
        """
        Convolutional block with optional batch normalization and dropout
        
        Args:
            inputs: Input tensor
            filters: Number of filters
            kernel_size: Size of convolutional kernel
            activation: Activation function
            batch_norm: Whether to apply batch normalization
            dropout_rate: Dropout rate (0 means no dropout)
            
        Returns:
            Output tensor after convolution operations
        """
        x = layers.Conv2D(filters, kernel_size, padding='same')(inputs)
        
        if batch_norm:
            x = layers.BatchNormalization()(x)
            
        x = layers.Activation(activation)(x)
        
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
            
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        
        if batch_norm:
            x = layers.BatchNormalization()(x)
            
        x = layers.Activation(activation)(x)
        
        return x
    
    def _encoder_block(self, inputs, filters, pool_size=2, dropout_rate=0.0):
        """
        Encoder block with convolution and max pooling
        
        Args:
            inputs: Input tensor
            filters: Number of filters
            pool_size: Size of max pooling
            dropout_rate: Dropout rate
            
        Returns:
            conv: Convolution output (for skip connection)
            pool: Pooled output (for next encoder block)
        """
        conv = self._conv_block(inputs, filters, dropout_rate=dropout_rate)
        pool = layers.MaxPooling2D(pool_size=pool_size)(conv)
        
        return conv, pool
    
    def _decoder_block(self, inputs, skip_connection, filters, kernel_size=2, 
                      dropout_rate=0.0):
        """
        Decoder block with upsampling and concatenation
        
        Args:
            inputs: Input tensor from previous layer
            skip_connection: Skip connection from encoder
            filters: Number of filters
            kernel_size: Size of transpose convolution kernel
            dropout_rate: Dropout rate
            
        Returns:
            Output tensor after upsampling and convolution
        """
        # Upsampling
        up = layers.Conv2DTranspose(filters, kernel_size, strides=2, 
                                   padding='same')(inputs)
        
        # Concatenate with skip connection
        concat = layers.Concatenate()([up, skip_connection])
        
        # Convolution block
        conv = self._conv_block(concat, filters, dropout_rate=dropout_rate)
        
        return conv
    
    def _build_model(self):
        """
        Build the complete U-Net model
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder path
        conv1, pool1 = self._encoder_block(inputs, self.filters, dropout_rate=0.1)
        conv2, pool2 = self._encoder_block(pool1, self.filters*2, dropout_rate=0.1)
        conv3, pool3 = self._encoder_block(pool2, self.filters*4, dropout_rate=0.2)
        conv4, pool4 = self._encoder_block(pool3, self.filters*8, dropout_rate=0.2)
        
        # Bottleneck
        conv5 = self._conv_block(pool4, self.filters*16, dropout_rate=0.3)
        
        # Decoder path
        conv6 = self._decoder_block(conv5, conv4, self.filters*8, dropout_rate=0.2)
        conv7 = self._decoder_block(conv6, conv3, self.filters*4, dropout_rate=0.2)
        conv8 = self._decoder_block(conv7, conv2, self.filters*2, dropout_rate=0.1)
        conv9 = self._decoder_block(conv8, conv1, self.filters, dropout_rate=0.1)
        
        # Output layer
        if self.num_classes == 1:
            # Binary segmentation
            outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
        else:
            # Multi-class segmentation
            outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(conv9)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='U-Net')
        
        return model
    
    def compile_model(self, optimizer='adam', loss='binary_crossentropy', 
                     metrics=['accuracy']):
        """
        Compile the model with specified optimizer, loss, and metrics
        
        Args:
            optimizer: Optimizer for training
            loss: Loss function
            metrics: List of metrics to track
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def dice_coefficient(self, y_true, y_pred, smooth=1e-6):
        """
        Dice coefficient for segmentation evaluation
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice coefficient value
        """
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        return dice
    
    def dice_loss(self, y_true, y_pred):
        """
        Dice loss function (1 - dice coefficient)
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            
        Returns:
            Dice loss value
        """
        return 1 - self.dice_coefficient(y_true, y_pred)
    
    def train(self, train_generator, validation_generator, epochs=50, 
              callbacks=None, verbose=1):
        """
        Train the U-Net model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            callbacks: List of Keras callbacks
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Compile with dice loss for better segmentation performance
        self.model.compile(
            optimizer='adam',
            loss=self.dice_loss,
            metrics=[self.dice_coefficient, 'accuracy']
        )
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    'best_unet_model.h5',
                    monitor='val_dice_coefficient',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
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
        Predict segmentation mask for input image
        
        Args:
            image: Input image array
            
        Returns:
            Predicted segmentation mask
        """
        # Ensure image has correct shape
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Normalize if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Predict
        prediction = self.model.predict(image)
        
        # Return binary mask
        if self.num_classes == 1:
            mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
        else:
            mask = np.argmax(prediction[0], axis=-1).astype(np.uint8)
        
        return mask
    
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
