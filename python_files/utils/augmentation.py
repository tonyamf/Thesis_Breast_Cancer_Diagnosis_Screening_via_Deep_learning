import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class MammographyAugmentation:
    """
    Advanced data augmentation techniques specifically designed for mammography images
    """
    
    def __init__(self, augmentation_probability=0.8):
        """
        Initialize augmentation pipeline
        
        Args:
            augmentation_probability: Probability of applying augmentations
        """
        self.augmentation_probability = augmentation_probability
        self.setup_albumentations_pipeline()
        self.setup_keras_pipeline()
    
    def setup_albumentations_pipeline(self):
        """Set up Albumentations augmentation pipeline"""
        self.albumentations_transform = A.Compose([
            # Geometric transformations
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.3
                ),
            ], p=0.7),
            
            # Intensity transformations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            ], p=0.6),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2),
            ], p=0.4),
            
            # Elastic transformations (important for medical images)
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            ),
            
            # Grid distortion
            A.GridDistortion(p=0.2),
            
            # Optical distortion
            A.OpticalDistortion(p=0.2),
        ], p=self.augmentation_probability)
    
    def setup_keras_pipeline(self):
        """Set up Keras ImageDataGenerator pipeline"""
        self.keras_generator = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='constant',
            cval=0
        )
    
    def apply_albumentations(self, image, mask=None):
        """
        Apply Albumentations transformations
        
        Args:
            image: Input image
            mask: Optional segmentation mask
            
        Returns:
            Augmented image and mask (if provided)
        """
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if mask is not None:
            # Apply transformations to both image and mask
            transformed = self.albumentations_transform(image=image, mask=mask)
            return transformed['image'], transformed['mask']
        else:
            # Apply transformations to image only
            transformed = self.albumentations_transform(image=image)
            return transformed['image']
    
    def apply_keras_augmentation(self, image):
        """
        Apply Keras augmentation
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Ensure image has correct shape for Keras
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Apply augmentation
        augmented = self.keras_generator.random_transform(image)
        
        return augmented
    
    def mixup_augmentation(self, images, labels, alpha=0.2):
        """
        Apply MixUp augmentation
        
        Args:
            images: Batch of images
            labels: Batch of labels
            alpha: MixUp parameter
            
        Returns:
            Mixed images and labels
        """
        batch_size = len(images)
        
        # Generate random lambda values
        lam = np.random.beta(alpha, alpha, batch_size)
        lam = np.maximum(lam, 1 - lam)  # Ensure lambda >= 0.5
        
        # Shuffle indices
        indices = np.random.permutation(batch_size)
        
        # Mix images
        mixed_images = []
        mixed_labels = []
        
        for i in range(batch_size):
            lam_i = lam[i]
            j = indices[i]
            
            # Mix images
            mixed_image = lam_i * images[i] + (1 - lam_i) * images[j]
            mixed_images.append(mixed_image)
            
            # Mix labels
            if len(labels.shape) > 1:  # One-hot encoded
                mixed_label = lam_i * labels[i] + (1 - lam_i) * labels[j]
            else:  # Integer labels
                mixed_label = labels[i] if np.random.random() < lam_i else labels[j]
            mixed_labels.append(mixed_label)
        
        return np.array(mixed_images), np.array(mixed_labels)
    
    def cutmix_augmentation(self, images, labels, alpha=1.0):
        """
        Apply CutMix augmentation
        
        Args:
            images: Batch of images
            labels: Batch of labels
            alpha: CutMix parameter
            
        Returns:
            CutMix augmented images and labels
        """
        batch_size, height, width = images.shape[:3]
        
        # Generate random lambda
        lam = np.random.beta(alpha, alpha)
        
        # Random indices for mixing
        indices = np.random.permutation(batch_size)
        
        # Calculate bounding box
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        
        # Random center point
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        
        # Bounding box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)
        
        # Apply CutMix
        mixed_images = images.copy()
        mixed_labels = []
        
        for i in range(batch_size):
            j = indices[i]
            
            # Cut and mix
            mixed_images[i, bby1:bby2, bbx1:bbx2] = images[j, bby1:bby2, bbx1:bbx2]
            
            # Adjust lambda based on actual cut area
            lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))
            
            # Mix labels
            if len(labels.shape) > 1:  # One-hot encoded
                mixed_label = lam_adjusted * labels[i] + (1 - lam_adjusted) * labels[j]
            else:  # Integer labels
                mixed_label = labels[i] if np.random.random() < lam_adjusted else labels[j]
            mixed_labels.append(mixed_label)
        
        return mixed_images, np.array(mixed_labels)
    
    def medical_specific_augmentation(self, image):
        """
        Apply medical imaging specific augmentations
        
        Args:
            image: Input medical image
            
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        # Random intensity variations (simulating different acquisition parameters)
        if np.random.random() < 0.3:
            # Simulate different exposure settings
            exposure_factor = np.random.uniform(0.8, 1.2)
            augmented = np.clip(augmented * exposure_factor, 0, 255)
        
        # Simulate different contrast settings
        if np.random.random() < 0.3:
            # Apply random gamma correction
            gamma = np.random.uniform(0.7, 1.3)
            augmented = np.power(augmented / 255.0, gamma) * 255.0
            augmented = np.clip(augmented, 0, 255)
        
        # Simulate acquisition noise
        if np.random.random() < 0.2:
            # Add Gaussian noise
            noise_std = np.random.uniform(1, 5)
            noise = np.random.normal(0, noise_std, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 255)
        
        # Simulate compression artifacts
        if np.random.random() < 0.1:
            # Simulate JPEG compression
            quality = np.random.randint(70, 95)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', augmented.astype(np.uint8), encode_param)
            augmented = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        
        return augmented.astype(np.uint8)
    
    def create_augmentation_pipeline(self, method='albumentations', include_mixup=False, 
                                   include_cutmix=False, include_medical=True):
        """
        Create a complete augmentation pipeline
        
        Args:
            method: Augmentation method ('albumentations' or 'keras')
            include_mixup: Whether to include MixUp augmentation
            include_cutmix: Whether to include CutMix augmentation
            include_medical: Whether to include medical-specific augmentations
            
        Returns:
            Augmentation function
        """
        def augment_fn(image, mask=None, label=None):
            # Apply base augmentations
            if method == 'albumentations':
                if mask is not None:
                    aug_image, aug_mask = self.apply_albumentations(image, mask)
                else:
                    aug_image = self.apply_albumentations(image)
                    aug_mask = None
            else:
                aug_image = self.apply_keras_augmentation(image)
                aug_mask = mask
            
            # Apply medical-specific augmentations
            if include_medical:
                aug_image = self.medical_specific_augmentation(aug_image)
            
            return aug_image, aug_mask, label
        
        return augment_fn
    
    def augment_dataset(self, images, labels, masks=None, augmentation_factor=2):
        """
        Augment an entire dataset
        
        Args:
            images: Array of images
            labels: Array of labels
            masks: Optional array of masks
            augmentation_factor: Factor by which to increase dataset size
            
        Returns:
            Augmented dataset
        """
        augmented_images = []
        augmented_labels = []
        augmented_masks = [] if masks is not None else None
        
        # Keep original data
        augmented_images.extend(images)
        augmented_labels.extend(labels)
        if masks is not None:
            augmented_masks.extend(masks)
        
        # Generate augmented samples
        for _ in range(augmentation_factor - 1):
            for i, image in enumerate(images):
                label = labels[i]
                mask = masks[i] if masks is not None else None
                
                # Apply augmentation
                if mask is not None:
                    aug_image, aug_mask = self.apply_albumentations(image, mask)
                    augmented_masks.append(aug_mask)
                else:
                    aug_image = self.apply_albumentations(image)
                
                # Apply medical-specific augmentation
                aug_image = self.medical_specific_augmentation(aug_image)
                
                augmented_images.append(aug_image)
                augmented_labels.append(label)
        
        result = {
            'images': np.array(augmented_images),
            'labels': np.array(augmented_labels)
        }
        
        if augmented_masks is not None:
            result['masks'] = np.array(augmented_masks)
        
        return result
    
    def tf_augmentation_pipeline(self):
        """
        Create TensorFlow augmentation pipeline
        
        Returns:
            TensorFlow augmentation function
        """
        @tf.function
        def augment_tf(image, label):
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            
            # Random rotation
            image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            
            # Random brightness and contrast
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            
            # Random saturation (if RGB)
            if image.shape[-1] == 3:
                image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
            
            # Ensure values are in valid range
            image = tf.clip_by_value(image, 0.0, 1.0)
            
            return image, label
        
        return augment_tf
