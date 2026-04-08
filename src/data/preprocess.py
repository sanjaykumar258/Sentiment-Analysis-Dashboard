import re
from typing import Any

import emoji
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)


class SocialMediaPreprocessor:
    """
    A class for preprocessing social media text data for sentiment analysis.
    Provides robust methods for cleaning mentions, URLs, hashtags, and emojis.
    """

    @staticmethod
    def clean_mentions(text: str) -> str:
        """
        Replaces @usernames with a generic @USER token.
        
        Args:
            text (str): The original text.
            
        Returns:
            str: Text with mentions replaced.
        """
        if not isinstance(text, str):
            return text
        return re.sub(r'@[A-Za-z0-9_]+', '@USER', text)

    @staticmethod
    def remove_urls(text: str) -> str:
        """
        Removes HTTP/HTTPS links from the text.
        
        Args:
            text (str): The text potentially containing URLs.
            
        Returns:
            str: Text with URLs removed.
        """
        if not isinstance(text, str):
            return text
        return re.sub(r'https?://\S+|www\.\S+', '', text)

    @staticmethod
    def normalize_hashtags(text: str) -> str:
        """
        Removes the # symbol but keeps the hashtag text.
        
        Args:
            text (str): The text with hashtags.
            
        Returns:
            str: Text with the # symbol removed from hashtags.
        """
        if not isinstance(text, str):
            return text
        return text.replace('#', '')

    @staticmethod
    def translate_emojis(text: str) -> str:
        """
        Converts emojis into text tokens using emoji.demojize.
        
        Args:
            text (str): Text containing emojis.
            
        Returns:
            str: Text with emojis translated to their textual representations.
        """
        if not isinstance(text, str):
            return text
        return emoji.demojize(text)

    @staticmethod
    def clean_single_text(text: Any) -> str:
        """
        Executes the full cleaning pipeline on a single text string.
        Safely handles NaN or non-string inputs.
        
        Args:
            text (Any): The input text to clean.
            
        Returns:
            str: The fully cleaned and stripped string.
        """
        if not isinstance(text, str):
            return ""

        text = SocialMediaPreprocessor.clean_mentions(text)
        text = SocialMediaPreprocessor.remove_urls(text)
        text = SocialMediaPreprocessor.normalize_hashtags(text)
        text = SocialMediaPreprocessor.translate_emojis(text)
        
        # Strip excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @classmethod
    def process_dataframe(cls, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Applies the cleaning pipeline to a specific column of a DataFrame.
        Uses list comprehensions for optimal performance.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            text_column (str): The name of the column containing the text to clean.
            
        Returns:
            pd.DataFrame: A new DataFrame with the cleaned text and empty rows dropped.
        """
        logger.info(f"Starting DataFrame preprocessing for column: '{text_column}' with {len(df)} rows.")
        
        # Ensure the column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the DataFrame.")
            
        # Create a copy to avoid SettingWithCopyWarning
        df_processed = df.copy()
        
        # List comprehension is much faster than apply() for strings
        cleaned_texts = [cls.clean_single_text(text) for text in df_processed[text_column]]
        
        # Assign the cleaned texts back to the column
        df_processed[text_column] = cleaned_texts
        
        # Filter out empty strings
        orig_len = len(df_processed)
        df_processed = df_processed[df_processed[text_column] != ""]
        dropped_count = orig_len - len(df_processed)
        
        logger.info(f"Preprocessing complete. Dropped {dropped_count} rows due to empty extracted text.")
        
        return df_processed


if __name__ == "__main__":
    # Small mock DataFrame containing fake Instagram comments
    mock_data = {
        "comment": [
            "Just got my new shoes! Thanks @nike for the fast delivery! #running #shoes 🏃‍♂️🔥",
            "This is the worst customer service ever... 😡 Check out my full review here: https://t.co/fakeurl",
            "I absolutely love this new feature! Great job @devteam 👏👏",
            "   ",  # Blank space, will be filtered out
            None  # NaN equivalent, will be filtered out
        ]
    }
    
    df_mock = pd.DataFrame(mock_data)
    
    print("--- ORIGINAL DATAFRAME ---")
    print(df_mock)
    print("\n--- PREPROCESSING ---")
    
    df_cleaned = SocialMediaPreprocessor.process_dataframe(df_mock, text_column="comment")
    
    print("\n--- CLEANED DATAFRAME ---")
    print(df_cleaned)
    for idx, row in df_cleaned.iterrows():
        print(f"[{idx}]: {row['comment']}")
