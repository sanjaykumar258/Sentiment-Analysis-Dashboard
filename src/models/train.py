import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from config.settings import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """
    Calculates Accuracy, F1-Score, Precision, and Recall using strictly scikit-learn.
    
    Args:
        eval_pred (EvalPrediction): A tuple containing model predictions and actual labels.
        
    Returns:
        dict: A dictionary containing the computed metrics.
    """
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


class SentimentTrainer:
    """
    Trainer class for fine-tuning the Hugging Face sentiment analysis model.
    Handles data preparation, training loops, and model checkpointing.
    """
    
    def __init__(self) -> None:
        """Initializes the trainer with global settings and logger."""
        self.settings = get_settings()
        logger.info("Initializing SentimentTrainer")
        self.dataset_dict: DatasetDict | None = None
        
    def prepare_data(self, csv_path: str) -> None:
        """
        Loads preprocessed CSV data, splits it into train/test sets, 
        and converts it into a Hugging Face DatasetDict structure.
        
        Args:
            csv_path (str): Path to the preprocessed data CSV.
        """
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if "text" not in df.columns or "label" not in df.columns:
            logger.warning("DataFrame must contain 'text' and 'label' columns for training.")
        
        # Split into train/test (80/20)
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
        
        self.dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
        logger.info(f"Prepared Data: {len(train_dataset)} train, {len(test_dataset)} test samples.")
        
    def train_model(self) -> None:
        """
        Tokenizes the dataset, initializes the Trainer, executes training, 
        and saves the best model to the configured directory.
        """
        if self.dataset_dict is None:
            raise ValueError("Data not prepared. Call prepare_data first.")
            
        model_name = self.settings.model.model_name
        logger.info(f"Loading tokenizer and model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Using num_labels=3 assuming Negative, Neutral, Positive scheme
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        
        max_length = self.settings.model.max_length

        def tokenize_function(examples: dict) -> dict:
            return tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=max_length
            )
            
        logger.info("Tokenizing datasets...")
        tokenized_datasets = self.dataset_dict.map(tokenize_function, batched=True)
        
        batch_size = self.settings.model.batch_size
        
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",  # evaluates validation set every epoch
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        
        logger.info("Starting model training execution...")
        trainer.train()
        
        save_dir = self.settings.paths.model_dir
        logger.info(f"Training complete. Saving best model and tokenizer to {save_dir}")
        
        # Critical execution requirement: Save best model to configured location
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info("Model saved successfully.")
