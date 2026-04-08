import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class InferencePipeline:
    """
    Pipeline for executing sentiment analysis prediction on text inputs.
    Features GPU acceleration, graceful fallback loading, and batching.
    """
    
    def __init__(self) -> None:
        """
        Initializes the model, moving it to the optimal available compute device.
        Attempts to load the local trained model, falling back to Hugging Face Hub if missing.
        """
        self.settings = get_settings()
        logger.info("Initializing InferencePipeline")
        
        # Optimal compute device placement
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        logger.info(f"Using compute device: {self.device}")
        
        model_dir = self.settings.paths.model_dir
        
        # Safe loading with fallback mechanisms
        try:
            # We strictly verify config.json exists before calling from_pretrained
            if not (model_dir / "config.json").exists():
                raise FileNotFoundError(f"Local config json missing in {model_dir}")
                
            logger.info(f"Loading local model and tokenizer from directory: {model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        except Exception as e:
            fallback_model = self.settings.model.model_name
            logger.warning(f"Failed to load local model from directory: {e}")
            logger.warning(f"Falling back to download default model from Hugging Face Hub: {fallback_model}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                fallback_model,
                num_labels=3,
                ignore_mismatched_sizes=True
            )
            
        # Move model to configured device and set into evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Typical 3-Class Sentiment Mapping configuration
        self.id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
        # Verify model specific labels dynamically just in case
        if hasattr(self.model.config, "id2label") and len(self.model.config.id2label) == 3:
            if 0 in self.model.config.id2label:
                self.id2label = self.model.config.id2label

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Runs inference smoothly mapping raw input text strings to probability distributions.
        
        Args:
            texts (list[str]): A batch list of text strings.
            
        Returns:
            list[dict]: Output dictionaries featuring predictions and probabilities.
        """
        if not texts:
            logger.error("Predict batch received an empty list of strings.")
            return []
            
        results = []
        max_length = self.settings.model.max_length
        
        try:
            # Tokenize and enforce truncation and padding across the batch
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Transfer tensor map efficiently to processing device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Execute inference explicitly without memory cost related to grad
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            logits = outputs.logits
            
            # Gather confidences using softmax 
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            
            for probs in probabilities:
                pred_idx = probs.argmax()
                predicted_label = self.id2label.get(pred_idx, str(pred_idx))
                
                # Format scoring 
                scores = {self.id2label.get(i, str(i)): float(prob) for i, prob in enumerate(probs)}
                
                results.append({
                    "predicted_label": predicted_label,
                    "confidence": float(probs[pred_idx]),
                    "class_probabilities": scores
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Execution error during predict_batch calculation: {e}")
            raise
