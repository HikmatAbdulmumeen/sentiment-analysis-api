import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import nltk

# DOWNLOAD REQUIRED NLTK DATA
@st.cache_resource
def load_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

load_nltk_data()

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

sia = SentimentIntensityAnalyzer()

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Page setup
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="centered")

st.title("😊 Sentiment Analysis Dashboard")
st.markdown("Analyze emotions in text using **VADER NLP** | Built by Hikmat Abdulmumeen")

# Text input
user_text = st.text_area("Enter your text here:", 
                         "He raped her. This is a terrible crime that must be punished severely.", 
                         height=150)

if st.button("Analyze Sentiment 🔍", type="primary"):
    if user_text.strip():
        # Analyze with VADER
        scores = sia.polarity_scores(user_text)
        
        # VADER gives: neg, neu, pos, compound (-1 to 1)
        compound = scores['compound']  # Overall sentiment
        neg = scores['neg']
        neu = scores['neu']
        pos = scores['pos']
        
        # Results columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Overall Sentiment")
            if compound >= 0.05:
                st.success(f"**Positive** 😊")
                emoji = "😊"
                label = f"Positive ({compound:.3f})"
            elif compound <= -0.05:
                st.error(f"**Negative** 😠")
                emoji = "😠"
                label = f"Negative ({compound:.3f})"
            else:
                st.info(f"**Neutral** 😐")
                emoji = "😐"
                label = f"Neutral ({compound:.3f})"
            
            st.write(f"**Compound Score:** {compound:.3f}")
            # Scale -1 to 1 → 0 to 1 for progress bar
            st.progress((compound + 1) / 2)
        
        with col2:
            st.subheader("📈 Score Breakdown")
            # Sentiment distribution
            st.write(f"**Positive:** {pos:.1%}")
            st.write(f"**Neutral:** {neu:.1%}")
            st.write(f"**Negative:** {neg:.1%}")
            
            # Pie chart
            if neg + pos > 0:  # Only show if not all neutral
                fig1, ax1 = plt.subplots(figsize=(4, 4))
                colors = ['#ff6b6b', '#51cf66', '#868e96']  # red, green, gray
                ax1.pie([neg, pos, neu], labels=['Negative', 'Positive', 'Neutral'],
                       colors=colors, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')
                st.pyplot(fig1)
        
        with col3:
            st.subheader("⚡ Intensity")
            intensity = abs(compound)
            if intensity > 0.5:
                st.warning(f"**High Intensity** 🔥")
                st.write("Strong emotional content")
            elif intensity > 0.2:
                st.info(f"**Medium Intensity** ⚡")
                st.write("Moderate emotion")
            else:
                st.success(f"**Low Intensity** 🌊")
                st.write("Subtle or mixed emotions")
            
            st.metric("Intensity Level", f"{intensity:.3f}")
        
        # Visual gauge
        st.subheader("🎯 Sentiment Gauge")
        fig2, ax2 = plt.subplots(figsize=(10, 2))
        
        # Create gauge
        ax2.barh([0], [compound], color='steelblue', height=0.5)
        ax2.set_xlim(-1, 1)
        ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax2.set_xticklabels(['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_yticks([])
        ax2.set_title(f"Overall: {emoji} (Score: {compound:.3f})")
        st.pyplot(fig2)
        
        # Details
        with st.expander("🔍 View Detailed Analysis"):
            st.write(f"**Original Text:** {user_text}")
            st.write(f"**Word Count:** {len(user_text.split())}")
            st.write("**VADER Scores:**")
            st.json(scores)
            
            # Show sentence-level analysis if multiple sentences
            sentences = nltk.sent_tokenize(user_text)
            if len(sentences) > 1:
                st.write("**Sentence-by-Sentence Analysis:**")
                for i, sent in enumerate(sentences):
                    sent_scores = sia.polarity_scores(sent)
                    st.write(f"{i+1}. **'{sent}'**")
                    st.write(f"   → Compound: {sent_scores['compound']:.3f}")
        
        # Example buttons
        st.subheader("💡 Try These Test Cases")
        examples = st.columns(4)
        test_cases = [
            "He raped her. This is horrific.",
            "I'm so happy today! 😊",
            "This is neither good nor bad.",
            "I HATE this with all my heart!"
        ]
        
        for i, col in enumerate(examples):
            with col:
                if st.button(f"Test {i+1}", key=f"ex{i}"):
                    st.session_state.text = test_cases[i]
                    st.rerun()
    else:
        st.warning("Please enter some text!")

# Footer
st.divider()
st.caption("Built with Streamlit & VADER NLP | Machine Learning Portfolio Project")
st.caption("👨‍💻 Created by Hikmat Abdulmumeen | Coursera ML Specialist")

