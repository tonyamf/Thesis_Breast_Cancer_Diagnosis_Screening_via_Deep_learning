import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers
import numpy as np
from sklearn.model_selection import train_test_split

class MetaPseudoLabels:
    """
    Meta Pseudo Labels implementation for semi-supervised learning
    Based on: "Meta Pseudo Labels" (https://arxiv.org/abs/2003.10580)
    
    This implementation uses a teacher-student framework where:
    - Teacher generates pseudo labels for unlabeled data
    - Student learns from both labeled and pseudo-labeled data
    - Teacher is updated based on student's performance on validation data
    """
    
    def __init__(self, teacher_model, student_model, num_classes=3, 
                 confidence_threshold=0.8, temperature=1.0):
        """
        Initialize Meta Pseudo Labels framework
        
        Args:
            teacher_model: Teacher network (pre-trained model)
            student_model: Student network (model to be trained)
            num_classes: Number of classes
            confidence_threshold: Minimum confidence for pseudo labels
            temperature: Temperature for softmax (controls sharpness)
        """
        self.teacher = teacher_model
        self.student = student_model
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        
        # Optimizers
        self.teacher_optimizer = optimizers.Adam(learning_rate=1e-4)
        self.student_optimizer = optimizers.Adam(learning_rate=1e-3)
        
        # Loss functions
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy()
        self.mse_loss = tf.keras.losses.MeanSquaredError()
    
    def temperature_scaling(self, logits, temperature):
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Model logits
            temperature: Temperature parameter
            
        Returns:
            Temperature-scaled probabilities
        """
        scaled_logits = logits / temperature
        return tf.nn.softmax(scaled_logits)
    
    def generate_pseudo_labels(self, unlabeled_data):
        """
        Generate pseudo labels using teacher model
        
        Args:
            unlabeled_data: Unlabeled input data
            
        Returns:
            pseudo_labels: Generated pseudo labels
            confidence_mask: Mask for high-confidence predictions
        """
        # Get teacher predictions
        teacher_logits = self.teacher(unlabeled_data, training=False)
        teacher_probs = self.temperature_scaling(teacher_logits, self.temperature)
        
        # Get confidence scores (max probability)
        confidence_scores = tf.reduce_max(teacher_probs, axis=1)
        
        # Create mask for high-confidence predictions
        confidence_mask = confidence_scores >= self.confidence_threshold
        
        # Generate pseudo labels (one-hot encoded)
        pseudo_labels = tf.one_hot(tf.argmax(teacher_probs, axis=1), self.num_classes)
        
        return pseudo_labels, confidence_mask
    
    @tf.function
    def student_training_step(self, labeled_data, labeled_labels, 
                             unlabeled_data, pseudo_labels, confidence_mask):
        """
        Single training step for student model
        
        Args:
            labeled_data: Labeled input data
            labeled_labels: True labels for labeled data
            unlabeled_data: Unlabeled input data
            pseudo_labels: Generated pseudo labels
            confidence_mask: Mask for high-confidence pseudo labels
            
        Returns:
            student_loss: Total loss for student
        """
        with tf.GradientTape() as tape:
            # Student predictions on labeled data
            labeled_logits = self.student(labeled_data, training=True)
            labeled_loss = self.cross_entropy(labeled_labels, labeled_logits)
            
            # Student predictions on unlabeled data
            unlabeled_logits = self.student(unlabeled_data, training=True)
            
            # Apply confidence mask to pseudo labels
            masked_pseudo_labels = tf.boolean_mask(pseudo_labels, confidence_mask)
            masked_unlabeled_logits = tf.boolean_mask(unlabeled_logits, confidence_mask)
            
            # Pseudo label loss (only for high-confidence predictions)
            if tf.shape(masked_pseudo_labels)[0] > 0:
                pseudo_loss = self.cross_entropy(masked_pseudo_labels, masked_unlabeled_logits)
            else:
                pseudo_loss = 0.0
            
            # Total student loss
            student_loss = labeled_loss + pseudo_loss
        
        # Update student parameters
        student_gradients = tape.gradient(student_loss, self.student.trainable_variables)
        self.student_optimizer.apply_gradients(
            zip(student_gradients, self.student.trainable_variables)
        )
        
        return student_loss
    
    @tf.function
    def teacher_training_step(self, validation_data, validation_labels, 
                             unlabeled_data, student_params_before):
        """
        Single training step for teacher model (meta-learning)
        
        Args:
            validation_data: Validation input data
            validation_labels: Validation labels
            unlabeled_data: Unlabeled data used for pseudo labeling
            student_params_before: Student parameters before update
            
        Returns:
            teacher_loss: Meta loss for teacher
        """
        with tf.GradientTape() as tape:
            # Generate pseudo labels with current teacher
            pseudo_labels, confidence_mask = self.generate_pseudo_labels(unlabeled_data)
            
            # Simulate student update (forward pass only for gradient computation)
            with tf.GradientTape() as student_tape:
                student_tape.watch(self.student.trainable_variables)
                
                # Student predictions on unlabeled data
                unlabeled_logits = self.student(unlabeled_data, training=True)
                
                # Apply confidence mask
                masked_pseudo_labels = tf.boolean_mask(pseudo_labels, confidence_mask)
                masked_unlabeled_logits = tf.boolean_mask(unlabeled_logits, confidence_mask)
                
                # Pseudo loss for student
                if tf.shape(masked_pseudo_labels)[0] > 0:
                    student_pseudo_loss = self.cross_entropy(masked_pseudo_labels, 
                                                           masked_unlabeled_logits)
                else:
                    student_pseudo_loss = 0.0
            
            # Get gradients for student update
            student_gradients = student_tape.gradient(student_pseudo_loss, 
                                                    self.student.trainable_variables)
            
            # Simulate student parameter update
            updated_student_params = []
            for param, grad in zip(self.student.trainable_variables, student_gradients):
                if grad is not None:
                    updated_param = param - self.student_optimizer.learning_rate * grad
                    updated_student_params.append(updated_param)
                else:
                    updated_student_params.append(param)
            
            # Temporarily set student parameters to updated values
            old_params = [param.numpy() for param in self.student.trainable_variables]
            for param, updated_param in zip(self.student.trainable_variables, updated_student_params):
                param.assign(updated_param)
            
            # Evaluate student on validation data
            val_logits = self.student(validation_data, training=False)
            teacher_loss = self.cross_entropy(validation_labels, val_logits)
            
            # Restore original student parameters
            for param, old_param in zip(self.student.trainable_variables, old_params):
                param.assign(old_param)
        
        # Update teacher parameters
        teacher_gradients = tape.gradient(teacher_loss, self.teacher.trainable_variables)
        self.teacher_optimizer.apply_gradients(
            zip(teacher_gradients, self.teacher.trainable_variables)
        )
        
        return teacher_loss
    
    def train_step(self, labeled_batch, unlabeled_batch, validation_batch):
        """
        Complete MPL training step
        
        Args:
            labeled_batch: Batch of labeled data (data, labels)
            unlabeled_batch: Batch of unlabeled data
            validation_batch: Batch of validation data (data, labels)
            
        Returns:
            Dictionary of losses
        """
        labeled_data, labeled_labels = labeled_batch
        validation_data, validation_labels = validation_batch
        
        # Store student parameters before update
        student_params_before = [param.numpy() for param in self.student.trainable_variables]
        
        # Generate pseudo labels
        pseudo_labels, confidence_mask = self.generate_pseudo_labels(unlabeled_batch)
        
        # Student training step
        student_loss = self.student_training_step(
            labeled_data, labeled_labels, 
            unlabeled_batch, pseudo_labels, confidence_mask
        )
        
        # Teacher training step (meta-learning)
        teacher_loss = self.teacher_training_step(
            validation_data, validation_labels,
            unlabeled_batch, student_params_before
        )
        
        return {
            'student_loss': student_loss,
            'teacher_loss': teacher_loss,
            'pseudo_label_ratio': tf.reduce_mean(tf.cast(confidence_mask, tf.float32))
        }
    
    def train(self, labeled_data, labeled_labels, unlabeled_data, 
              validation_data, validation_labels, epochs=50, batch_size=32):
        """
        Train the MPL framework
        
        Args:
            labeled_data: Labeled training data
            labeled_labels: Labels for training data
            unlabeled_data: Unlabeled training data
            validation_data: Validation data
            validation_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Convert labels to one-hot if needed
        if len(labeled_labels.shape) == 1:
            labeled_labels = tf.one_hot(labeled_labels, self.num_classes)
        if len(validation_labels.shape) == 1:
            validation_labels = tf.one_hot(validation_labels, self.num_classes)
        
        # Create datasets
        labeled_dataset = tf.data.Dataset.from_tensor_slices((labeled_data, labeled_labels))
        labeled_dataset = labeled_dataset.batch(batch_size).shuffle(1000)
        
        unlabeled_dataset = tf.data.Dataset.from_tensor_slices(unlabeled_data)
        unlabeled_dataset = unlabeled_dataset.batch(batch_size).shuffle(1000)
        
        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels))
        validation_dataset = validation_dataset.batch(batch_size)
        
        # Training history
        history = {
            'student_loss': [],
            'teacher_loss': [],
            'pseudo_label_ratio': [],
            'validation_accuracy': []
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_student_losses = []
            epoch_teacher_losses = []
            epoch_pseudo_ratios = []
            
            # Iterate through batches
            for (labeled_batch, unlabeled_batch, val_batch) in zip(
                labeled_dataset, unlabeled_dataset.repeat(), validation_dataset.repeat()
            ):
                # Training step
                losses = self.train_step(labeled_batch, unlabeled_batch, val_batch)
                
                epoch_student_losses.append(losses['student_loss'])
                epoch_teacher_losses.append(losses['teacher_loss'])
                epoch_pseudo_ratios.append(losses['pseudo_label_ratio'])
            
            # Calculate epoch averages
            avg_student_loss = np.mean(epoch_student_losses)
            avg_teacher_loss = np.mean(epoch_teacher_losses)
            avg_pseudo_ratio = np.mean(epoch_pseudo_ratios)
            
            # Validation accuracy
            val_accuracy = self.evaluate_student(validation_data, validation_labels)
            
            # Store history
            history['student_loss'].append(avg_student_loss)
            history['teacher_loss'].append(avg_teacher_loss)
            history['pseudo_label_ratio'].append(avg_pseudo_ratio)
            history['validation_accuracy'].append(val_accuracy)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Student Loss: {avg_student_loss:.4f}")
            print(f"Teacher Loss: {avg_teacher_loss:.4f}")
            print(f"Pseudo Label Ratio: {avg_pseudo_ratio:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print("-" * 50)
        
        return history
    
    def evaluate_student(self, test_data, test_labels):
        """
        Evaluate student model accuracy
        
        Args:
            test_data: Test input data
            test_labels: Test labels
            
        Returns:
            Accuracy score
        """
        predictions = self.student(test_data, training=False)
        predicted_classes = tf.argmax(predictions, axis=1)
        
        if len(test_labels.shape) > 1:
            true_classes = tf.argmax(test_labels, axis=1)
        else:
            true_classes = test_labels
        
        accuracy = tf.reduce_mean(tf.cast(predicted_classes == true_classes, tf.float32))
        return accuracy.numpy()
    
    def save_models(self, teacher_path, student_path):
        """
        Save teacher and student models
        
        Args:
            teacher_path: Path to save teacher model
            student_path: Path to save student model
        """
        self.teacher.save_weights(teacher_path)
        self.student.save_weights(student_path)
    
    def load_models(self, teacher_path, student_path):
        """
        Load teacher and student models
        
        Args:
            teacher_path: Path to teacher model weights
            student_path: Path to student model weights
        """
        self.teacher.load_weights(teacher_path)
        self.student.load_weights(student_path)
