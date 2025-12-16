import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Page setup
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="centered")

st.title("😊 Sentiment Analysis Dashboard")
st.markdown("Analyze emotions in text using **machine learning** | Built by Hikmat Abdulmumeen")

# Text input
user_text = st.text_area("Enter your text here:", "I absolutely love machine learning! It's amazing!", height=150)

if st.button("Analyze Sentiment 🔍", type="primary"):
    if user_text.strip():
        # Analyze
        blob = TextBlob(user_text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Results columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment")
            if polarity > 0.1:
                st.success(f"**Positive** 😊")
                emoji = "😊"
            elif polarity < -0.1:
                st.error(f"**Negative** 😠")
                emoji = "😠"
            else:
                st.info(f"**Neutral** 😐")
                emoji = "😐"
            
            # Score with color bar
            st.write(f"**Score:** {polarity:.3f}")
            st.progress((polarity + 1) / 2)
            
        with col2:
            st.subheader("📝 Subjectivity")
            if subjectivity > 0.6:
                st.info(f"**Opinionated** 💬")
            else:
                st.success(f"**Factual** 📊")
            
            st.write(f"**Score:** {subjectivity:.3f}")
            st.progress(subjectivity)
        
        # Visual gauge
        st.subheader("🎯 Sentiment Gauge")
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Create gauge
        ax.barh([0], [polarity], color='steelblue', height=0.5)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_yticks([])
        ax.set_title(f"Sentiment: {emoji} ({polarity:.3f})")
        st.pyplot(fig)
        
        # Details
        with st.expander("View Details"):
            st.write(f"**Original Text:** {user_text}")
            st.write(f"**Word Count:** {len(user_text.split())}")
            st.write(f"**Sentence Count:** {len(blob.sentences)}")
            st.json({
                "polarity": float(polarity),
                "subjectivity": float(subjectivity),
                "analysis": "positive" if polarity > 0 else "negative"
            })
        
        # Example for different texts
        st.subheader("💡 Try These Examples")
        examples = st.columns(3)
        example_texts = [
            "This product is terrible!",
            "The service was okay.",
            "I'm extremely happy!"
        ]
        
        for i, col in enumerate(examples):
            with col:
                if st.button(f"Example {i+1}", key=f"ex{i}"):
                    st.session_state.text = example_texts[i]
                    st.rerun()
    else:
        st.warning("Please enter some text!")

# Footer
st.divider()
st.caption("Built with Streamlit & TextBlob | Machine Learning Portfolio Project")

st.caption("👨‍💻 Created by Hikmat Abdulmumeen | Coursera ML Specialist")
