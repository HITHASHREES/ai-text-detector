import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModel

st.title("AI vs Human Text Detector")

@st.cache_resource
def load_all():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert = AutoModel.from_pretrained("distilbert-base-uncased")
    model = joblib.load("model.joblib")
    return tokenizer, bert, model

tokenizer, bert, model = load_all()

def embed(text):
    tokens = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = bert(**tokens)
    return out.last_hidden_state.mean(dim=1).numpy()

text = st.text_area("Paste text here")

if st.button("Detect"):
    vec = embed(text)
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]

    if pred == 1:
        st.error(f"AI Generated ({prob*100:.2f}%)")
    else:
        st.success(f"Human Written ({(1-prob)*100:.2f}%)")
