import streamlit as st
from io import BytesIO
import base64
import os

st.set_page_config(page_title="AI-powered News Brief Generator", layout="wide")

st.title("AI-powered news brief generator")
st.write("Quickly summarize news articles with sentiment analysis.")

# Utility functions
def try_transformers_summarize(text, max_length=60, min_length=30):
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        raise e

def gensim_summarize(text, word_count=50):
    try:
        from gensim.summarization import summarize
        return summarize(text, word_count=word_count)
    except Exception:
        raise

def freq_based_summarize(text, word_count=50):
    import re
    from collections import Counter
    from heapq import nlargest
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]
    if not sentences:
        return ""
    words = re.findall(r'\w+', text.lower())
    freq = Counter(words)
    maxf = max(freq.values())
    for k in freq:
        freq[k] = freq[k] / maxf
    sentence_scores = {}
    for s in sentences:
        s_words = re.findall(r'\w+', s.lower())
        score = sum(freq.get(w, 0) for w in s_words)
        sentence_scores[s] = score / (len(s_words)+1)
    ranked = nlargest(len(sentences), sentence_scores, key=sentence_scores.get)
    summary = []
    wc = 0
    for s in ranked:
        summary.append(s)
        wc += len(s.split())
        if wc >= word_count:
            break
    return ' '.join(summary[:3])

# Sentiment functions
def try_transformers_sentiment(text):
    try:
        from transformers import pipeline
        sentiment = pipeline("sentiment-analysis")
        out = sentiment(text[:512])
        return out[0]['label'], out[0].get('score', None)
    except Exception:
        raise

def vader_sentiment(text):
    try:
        import nltk
        nltk.download('vader_lexicon')
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(text)
        compound = score['compound']
        if compound >= 0.05:
            return "Positive", compound
        elif compound <= -0.05:
            return "Negative", compound
        else:
            return "Neutral", compound
    except Exception:
        raise

_positive_words = set(["good","great","positive","growth","up","increase","benefit","improve","improved","win","success","beneficial"])
_negative_words = set(["bad","decline","down","loss","negative","drop","fail","failure","concern","problem","risk"])
def simple_lexicon_sentiment(text):
    words = [w.strip(".,!?:;()\"'").lower() for w in text.split()]
    pos = sum(1 for w in words if w in _positive_words)
    neg = sum(1 for w in words if w in _negative_words)
    if pos > neg:
        return "Positive", (pos - neg) / (len(words)+1)
    elif neg > pos:
        return "Negative", (neg - pos) / (len(words)+1)
    else:
        return "Neutral", 0.0

# Main UI
with st.container():
    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("Enter or paste your news article text here...")
        text = st.text_area("", height=300, placeholder="Paste full news article here...")
        st.markdown("**Select Summary Length:**")
        length_choice = st.radio("", ("50 words", "100 words", "150 words"), index=0, horizontal=True)
        target_words = int(length_choice.split()[0])
        generate = st.button("Generate Brief", type="primary")

    with cols[1]:
        st.write("‎")
        st.info("Summary length options: choose how long you want the brief to be.")

if generate and text.strip():
    with st.spinner("Generating summary..."):
        summary = ""
        try:
            max_len = max(30, int(target_words * 1.6))
            min_len = max(10, int(target_words * 0.6))
            summary = try_transformers_summarize(text, max_length=max_len, min_length=min_len)
        except Exception:
            try:
                summary = gensim_summarize(text, word_count=target_words)
            except Exception:
                summary = freq_based_summarize(text, word_count=target_words)
        if not summary:
            summary = ' '.join(text.split()[:target_words]) + ('...' if len(text.split())>target_words else '')

    sentiment_label = "Neutral"
    sentiment_score = None
    try:
        sentiment_label, sentiment_score = try_transformers_sentiment(summary)
    except Exception:
        try:
            sentiment_label, sentiment_score = vader_sentiment(summary)
        except Exception:
            sentiment_label, sentiment_score = simple_lexicon_sentiment(summary)

    left, right = st.columns([3,1])
    with left:
        st.subheader("Generated Summary")
        st.write(summary)
        st.download_button("Download Summary", data=summary, file_name="summary.txt", mime="text/plain")
    with right:
        st.subheader("Sentiment Analysis")
        color = "gray"
        if sentiment_label.lower().startswith('pos'):
            color = "green"
        elif sentiment_label.lower().startswith('neg'):
            color = "red"
        st.markdown(f"<div style='padding:12px;border-radius:6px;background:{color};color:white;text-align:center'>{sentiment_label}</div>", unsafe_allow_html=True)
        if sentiment_score is not None:
            st.caption(f"Score: {sentiment_score:.3f}")

    st.success("Done. You can copy or download the summary.")
elif generate and not text.strip():
    st.error("Please paste a news article first to generate a summary.")

st.markdown("---")
st.caption("This app uses PEGASUS or BART for summarization if available. If not, lighter extractive methods are used as fallback.")
