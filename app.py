import streamlit as st
from textblob import TextBlob

st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")

st.title("😊 Sentiment Analysis App")
st.write("Enter text to analyze its sentiment")

# Input
text = st.text_area("Enter text:", "I love machine learning!")

if st.button("Analyze Sentiment"):
    if text.strip():
        # Analyze
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sentiment Score", f"{sentiment_score:.2f}")
            if sentiment_score > 0:
                st.success("Positive 😊")
            elif sentiment_score < 0:
                st.error("Negative 😠")
            else:
                st.info("Neutral 😐")
        
        with col2:
            st.metric("Subjectivity", f"{subjectivity:.2f}")
            if subjectivity > 0.5:
                st.write("Opinionated 💬")
            else:
                st.write("Factual 📊")
        
        # Details
        with st.expander("Detailed Analysis"):
            st.write(f"**Text:** {text}")
            st.write(f"**Polarity:** {sentiment_score:.3f} (-1 to 1)")
            st.write(f"**Subjectivity:** {subjectivity:.3f} (0=factual, 1=opinion)")
            
            # Sentiment bar
            st.progress((sentiment_score + 1) / 2)
    else:
        st.warning("Please enter some text")

# Instructions
st.sidebar.header("How to Use")
st.sidebar.write("""
1. Enter text in the box
2. Click 'Analyze Sentiment'
3. View results
""")
st.sidebar.header("About")
st.sidebar.write("""
Uses TextBlob for sentiment analysis.
- Polarity: -1 (negative) to 1 (positive)
- Subjectivity: 0 (objective) to 1 (subjective)
""")