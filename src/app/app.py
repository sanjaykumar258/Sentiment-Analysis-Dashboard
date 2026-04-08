import sys
from pathlib import Path

# Ensure the root directory is in the Python path
# so that imports from config, src, and utils resolve correctly.
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import plotly.express as px
import streamlit as st
from config.settings import get_settings
from src.data.preprocess import SocialMediaPreprocessor
from src.models.predict import InferencePipeline
from utils.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

st.set_page_config(
    page_title="Social Media Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model() -> InferencePipeline:
    """
    Instantiates the inference pipeline and caches it aggressively via st.cache_resource
    to prevent reloading the heavy Transformer model repeatedly on state updates.

    Returns:
        InferencePipeline: Singleton predictive pipeline for model inference.
    """
    logger.info("Initializing cached InferencePipeline for Streamlit.")
    return InferencePipeline()

def main():
    """Main rendering loop for the Streamlit dashboard app."""
    # Ensure model is ready quietly in background
    pipeline = load_model()

    # Layout: Sidebar Configuration
    st.sidebar.title("Social Media Sentiment Dashboard")
    st.sidebar.markdown(
        """
        Welcome to the sentiment analysis toolkit. 
        Leverage RoBERTa powered NLP to decode emotions in text precisely 
        and monitor broad social sentiment trends.
        """
    )
    
    st.sidebar.divider()
    app_mode = st.sidebar.radio("Navigation", ["Live Analysis", "Batch Processing"])

    # App Mode 1: Live Analysis
    if app_mode == "Live Analysis":
        st.title("Live Text Sentiment Analysis")
        st.write("Determine the exact emotion behind a single tweet or comment.")
        
        user_input = st.text_area("Enter social media text here:", height=150)
        
        if st.button("Analyze", type="primary"):
            if not user_input.strip():
                st.warning("Please provide some text to analyze.")
                return

            with st.spinner("Processing text..."):
                # Clean specifically for social media elements
                clean_text = SocialMediaPreprocessor.clean_single_text(user_input)
                
                # Fetch prediction output dictionary
                predictions = pipeline.predict_batch([clean_text])
                
                if predictions:
                    result = predictions[0]
                    predicted_label = result["predicted_label"]
                    confidence = result["confidence"]
                    scores_dict = result["class_probabilities"]

                    st.markdown("### Results")
                    
                    # Highlight colors depending on outcome logic
                    metric_color = "black"
                    if predicted_label.lower() == "positive":
                        metric_color = "green"
                    elif predicted_label.lower() == "negative":
                        metric_color = "red"
                    elif predicted_label.lower() == "neutral":
                        metric_color = "gray"
                        
                    st.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                            <h2 style="color: {metric_color}; margin: 0; padding: 0;">{predicted_label}</h2>
                            <p style="margin: 0; padding: 0;">Confidence Score: {confidence:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.divider()
                    st.markdown("#### Class Confidence Probabilities")

                    # Structuring a small dataframe purely for Plotly visualization
                    df_probs = pd.DataFrame({
                        "Class": list(scores_dict.keys()),
                        "Probability": list(scores_dict.values())
                    })
                    
                    # Plot interactive horizontal bar chart for the probabilities
                    fig = px.bar(
                        df_probs, 
                        x="Probability", 
                        y="Class", 
                        orientation='h',
                        text_auto='.1%',
                        color="Class",
                        color_discrete_map={
                            "Positive": "#28a745",
                            "Neutral": "#6c757d",
                            "Negative": "#dc3545"
                        }
                    )
                    fig.update_layout(showlegend=False, xaxis_title="Confidence Probability", yaxis_title="Sentiment Class")
                    st.plotly_chart(fig, use_container_width=True)

    # App Mode 2: Batch Processing
    elif app_mode == "Batch Processing":
        st.title("Batch Data Processing")
        st.write("Upload a CSV file holding mass sentiment records to enrich it with model predictions automatically.")
        
        uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
        
        if uploaded_file is not None:
            # Read CSV efficiently
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Request user select relevant text column for inference
            text_column = st.selectbox("Select the column containing the text to analyze:", df.columns)
            
            if st.button("Run Batch Analysis", type="primary"):
                with st.spinner(f"Cleaning text and running model on {len(df)} rows..."):
                    try:
                        # Utilize strict vectorised style processing explicitly
                        processed_df = SocialMediaPreprocessor.process_dataframe(df, text_column)
                        
                        if processed_df.empty:
                            st.error("No valid text remaining after preprocessing.")
                            return
                            
                        texts_to_predict = processed_df[text_column].tolist()
                        batch_results = pipeline.predict_batch(texts_to_predict)
                        
                        # Extract predictions back into the cleaned dataframe respectively
                        processed_df['predicted_sentiment'] = [res['predicted_label'] for res in batch_results]
                        processed_df['confidence'] = [res['confidence'] for res in batch_results]
                        
                        st.success(f"Successfully processed {len(processed_df)} valid textual entries.")
                        
                        # Tabs structure rendering visualizations nicely alongside textual outputs
                        tab1, tab2 = st.tabs(["Analytics visualization", "Data View & Export"])
                        
                        with tab1:
                            st.markdown("### Overall Sentiment Distribution")
                            sentiment_counts = processed_df['predicted_sentiment'].value_counts().reset_index()
                            sentiment_counts.columns = ['Sentiment', 'Total Records']
                            
                            # Interactive Pie Chart Visualization
                            fig_pie = px.pie(
                                sentiment_counts, 
                                names="Sentiment", 
                                values="Total Records", 
                                hole=0.4,
                                color="Sentiment",
                                color_discrete_map={
                                    "Positive": "#28a745",
                                    "Neutral": "#6c757d",
                                    "Negative": "#dc3545"
                                }
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                        with tab2:
                            st.markdown("### Finished Annotations")
                            st.dataframe(processed_df)
                            
                            csv_output = processed_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Annotated Predictions (CSV)",
                                data=csv_output,
                                file_name="sentiment_predictions_annotated.csv",
                                mime="text/csv",
                                type="primary"
                            )
                    except Exception as e:
                        st.error(f"Failed to process batch: {e}")
                        logger.error(f"Batch processing error in Streamlit: {e}")


if __name__ == "__main__":
    main()
