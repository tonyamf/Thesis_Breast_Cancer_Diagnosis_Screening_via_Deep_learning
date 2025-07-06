import cv2
import numpy as np
from skimage import filters, morphology, exposure
from scipy import ndimage
import tensorflow as tf

class MammographyPreprocessor:
    """
    Comprehensive preprocessing pipeline for mammography images
    Based on the research paper methodology
    """
    
    def __init__(self):
        """Initialize the preprocessor with default parameters"""
        self.default_size = (512, 512)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def remove_dicom_tags(self, image):
        """
        Remove DICOM tags and annotations from mammography images
        
        Args:
            image: Input mammography image
            
        Returns:
            Cleaned image without tags
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create a copy to work with
        cleaned_image = image.copy()
        
        # Remove bright text regions (typical DICOM tags)
        # Threshold to find bright regions
        _, binary = cv2.threshold(cleaned_image, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Remove small bright regions (likely text)
            if area < 5000:  # Adjust threshold based on image size
                cv2.fillPoly(cleaned_image, [contour], 0)
        
        return cleaned_image
    
    def remove_artifacts(self, image):
        """
        Remove scanning artifacts and labels
        
        Args:
            image: Input image
            
        Returns:
            Image with artifacts removed
        """
        # Remove edge artifacts
        h, w = image.shape[:2]
        
        # Create mask to remove border artifacts
        mask = np.ones_like(image, dtype=np.uint8)
        border_size = min(h, w) // 20  # Adaptive border size
        
        # Set border regions to zero
        mask[:border_size, :] = 0
        mask[-border_size:, :] = 0
        mask[:, :border_size] = 0
        mask[:, -border_size:] = 0
        
        # Apply mask
        cleaned_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Remove high-intensity outliers (scanning artifacts)
        threshold = np.percentile(image[image > 0], 99)
        cleaned_image[cleaned_image > threshold] = threshold
        
        return cleaned_image
    
    def noise_removal(self, image, method='gaussian_filter'):
        """
        Apply noise removal techniques
        
        Args:
            image: Input image
            method: Noise removal method ('gaussian_filter', 'median_filter', 'bilateral_filter')
            
        Returns:
            Denoised image
        """
        if method == 'gaussian_filter':
            # Gaussian filtering for noise reduction
            denoised = cv2.GaussianBlur(image, (5, 5), 1.0)
            
        elif method == 'median_filter':
            # Median filtering for salt-and-pepper noise
            denoised = cv2.medianBlur(image, 5)
            
        elif method == 'bilateral_filter':
            # Bilateral filtering to preserve edges while reducing noise
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
        else:
            raise ValueError(f"Unknown noise removal method: {method}")
        
        return denoised
    
    def enhance_contrast(self, image, method='clahe'):
        """
        Enhance image contrast
        
        Args:
            image: Input image
            method: Enhancement method ('clahe', 'histogram_equalization', 'gamma_correction')
            
        Returns:
            Enhanced image
        """
        if method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            enhanced = self.clahe.apply(image)
            
        elif method == 'histogram_equalization':
            # Global histogram equalization
            enhanced = cv2.equalizeHist(image)
            
        elif method == 'gamma_correction':
            # Gamma correction for brightness adjustment
            gamma = 0.8  # Adjust gamma value based on dataset
            enhanced = np.power(image / 255.0, gamma) * 255.0
            enhanced = enhanced.astype(np.uint8)
            
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
        
        return enhanced
    
    def morphological_operations(self, image):
        """
        Apply morphological operations to clean up the image
        
        Args:
            image: Input binary or grayscale image
            
        Returns:
            Processed image
        """
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Opening operation (erosion followed by dilation)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Closing operation (dilation followed by erosion)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def normalize_intensity(self, image):
        """
        Normalize image intensity values
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        # Convert to float
        image_float = image.astype(np.float32)
        
        # Normalize to [0, 1] range
        image_min = np.min(image_float)
        image_max = np.max(image_float)
        
        if image_max > image_min:
            normalized = (image_float - image_min) / (image_max - image_min)
        else:
            normalized = image_float
        
        return normalized
    
    def resize_image(self, image, target_size=None):
        """
        Resize image to target size while maintaining aspect ratio
        
        Args:
            image: Input image
            target_size: Target size tuple (height, width)
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.default_size
        
        # Get current dimensions
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate aspect ratio
        aspect_ratio = w / h
        target_aspect_ratio = target_w / target_h
        
        # Resize maintaining aspect ratio
        if aspect_ratio > target_aspect_ratio:
            # Image is wider, fit to width
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:
            # Image is taller, fit to height
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Pad to exact target size
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        padded = cv2.copyMakeBorder(
            resized, pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=0
        )
        
        return padded
    
    def segment_breast_region(self, image):
        """
        Segment the breast region from the background
        
        Args:
            image: Input mammography image
            
        Returns:
            Segmented breast region mask
        """
        # Apply threshold to separate breast tissue from background
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        
        # Find the largest connected component (breast region)
        largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        breast_mask = (labels == largest_component).astype(np.uint8) * 255
        
        # Apply morphological operations to smooth the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, kernel)
        breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_OPEN, kernel)
        
        return breast_mask
    
    def preprocess_image(self, image, apply_noise_removal=True, 
                        noise_method='gaussian_filter', apply_enhancement=True,
                        enhancement_method='clahe', apply_normalization=True,
                        target_size=(512, 512)):
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input mammography image
            apply_noise_removal: Whether to apply noise removal
            noise_method: Noise removal method
            apply_enhancement: Whether to apply contrast enhancement
            enhancement_method: Enhancement method
            apply_normalization: Whether to normalize intensity
            target_size: Target image size
            
        Returns:
            Preprocessed image
        """
        # Ensure image is in correct format
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        processed_image = image.copy()
        
        # Step 1: Remove DICOM tags and artifacts
        processed_image = self.remove_dicom_tags(processed_image)
        processed_image = self.remove_artifacts(processed_image)
        
        # Step 2: Noise removal
        if apply_noise_removal:
            processed_image = self.noise_removal(processed_image, method=noise_method)
        
        # Step 3: Contrast enhancement
        if apply_enhancement:
            processed_image = self.enhance_contrast(processed_image, method=enhancement_method)
        
        # Step 4: Morphological operations
        processed_image = self.morphological_operations(processed_image)
        
        # Step 5: Resize to target size
        processed_image = self.resize_image(processed_image, target_size)
        
        # Step 6: Intensity normalization
        if apply_normalization:
            processed_image = self.normalize_intensity(processed_image)
        
        return processed_image
    
    def batch_preprocess(self, image_batch, **kwargs):
        """
        Preprocess a batch of images
        
        Args:
            image_batch: Batch of images
            **kwargs: Preprocessing parameters
            
        Returns:
            Batch of preprocessed images
        """
        processed_batch = []
        
        for image in image_batch:
            processed_image = self.preprocess_image(image, **kwargs)
            processed_batch.append(processed_image)
        
        return np.array(processed_batch)
    
    def create_preprocessing_pipeline(self, **kwargs):
        """
        Create a TensorFlow preprocessing pipeline
        
        Args:
            **kwargs: Preprocessing parameters
            
        Returns:
            TensorFlow preprocessing function
        """
        def preprocess_fn(image):
            # Convert tensor to numpy for processing
            image_np = image.numpy()
            processed = self.preprocess_image(image_np, **kwargs)
            return tf.convert_to_tensor(processed, dtype=tf.float32)
        
        return tf.py_function(preprocess_fn, [tf.float32], tf.float32)
