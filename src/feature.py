import torch
from typing import Union, Optional
import logging
from PIL import Image
from transformers import (
    AutoFeatureExtractor, 
    AutoModelForImageClassification, 
    PreTrainedFeatureExtractor, 
    PreTrainedModel
)

class FeatureExtractor:
    """
    Advanced feature extraction class using pretrained deep learning models.
    
    Supports extracting high-level image features with flexible model configurations.
    """
    
    def _init_(
        self, 
        model_name: str = "microsoft/resnet-50",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize feature extractor with specified pretrained model.
        
        Args:
            model_name (str): Pretrained model identifier from Hugging Face model hub.
            logger (Optional[logging.Logger]): Custom logger for tracking operations.
        
        Raises:
            ValueError: If model cannot be loaded
        """
        self.logger = logger or logging.getLogger(_name_)
        
        try:
            self.feature_extractor: PreTrainedFeatureExtractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model: PreTrainedModel = AutoModelForImageClassification.from_pretrained(model_name)
            
            self.logger.info(f"Loaded feature extractor: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise ValueError(f"Model initialization failed: {e}")
    
    def preprocess_image(
        self, 
        image_path: str, 
        target_size: Optional[tuple] = None
    ) -> Image.Image:
        """
        Preprocess image with optional resizing.
        
        Args:
            image_path (str): Path to input image
            target_size (Optional[tuple]): Desired image dimensions (width, height)
        
        Returns:
            PIL.Image: Preprocessed image
        """
        image = Image.open(image_path).convert('RGB')
        
        if target_size:
            image = image.resize(target_size, Image.LANCZOS)
        
        return image
    
    def extract_features(
        self, 
        image_path: str, 
        target_size: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        Extract high-level features from an image.
        
        Args:
            image_path (str): Path to input image
            target_size (Optional[tuple]): Optional image resize dimensions
        
        Returns:
            torch.Tensor: Extracted feature tensor
        
        Raises:
            IOError: If image cannot be processed
            ValueError: If feature extraction fails
        """
        try:
            # Preprocess image
            image = self.preprocess_image(image_path, target_size)
            
            # Prepare inputs
            inputs = self.feature_extractor(image, return_tensors="pt")
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Log successful extraction
            self.logger.info(f"Successfully extracted features from {image_path}")
            
            return outputs.logits
        
        except IOError as e:
            self.logger.error(f"Image processing error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise ValueError(f"Feature extraction error: {e}")
    
    def get_top_k_predictions(
        self, 
        features: torch.Tensor, 
        k: int = 5
    ) -> list:
        """
        Get top-k predictions from extracted features.
        
        Args:
            features (torch.Tensor): Extracted feature tensor
            k (int): Number of top predictions to return
        
        Returns:
            list: Top-k predictions with probabilities
        """
        probabilities = torch.softmax(features, dim=1)
        top_k = torch.topk(probabilities, k)
        
        return [
            {
                "class_id": idx.item(), 
                "probability": prob.item()
            } 
            for idx, prob in zip(top_k.indices[0], top_k.values[0])
        ]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)