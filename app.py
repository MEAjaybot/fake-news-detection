import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import re
import sklearn
import joblib
from gensim.models import word2vec
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
from transformers import pipeline
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


tfidf= joblib.load("model/tfidf_vectorizer.joblib")
lr =  joblib.load("model/logistic_regression.pkl")


dimension = 100
def modelw2v(ind=dimension):
        return  nn.Sequential(nn.Linear(ind,75),
                         nn.Linear(75,75),
                         nn.ReLU(),
                         nn.Linear(75,50),
                         nn.ReLU(),
                         nn.Linear(50,25),
                         nn.ReLU(),
                         nn.Linear(25,10),
                         nn.ReLU(),
                         nn.Linear(10,5),
                         nn.ReLU(),
                         nn.Linear(5,1)
)
            


@st.cache_resource
def w2v():
    return word2vec.load("model/word2vec.model")

@st.cache_resource
def summm():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = summm()
@st.cache_resource
def load_w2v(ind=dimension):
        model = modelw2v(ind)
        state = torch.load("model/modelw2v.pth")
        model.load_state_dict(state)
        model.eval()
        return model

W2v = w2v()
model = load_w2v()

stop_words = set(stopwords.words("english"))
lem = WordNetLemmatizer()

def clear(text):
    text = BeautifulSoup(text,"html.parser").get_text()
    text = contractions.fix(text)
    text = re.sub(r'[^A-Z a-z\s]'," ", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t)>1]
    tokens = [lem.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


def vecsent(tokens):
    vectors=[]
    for word in tokens:
        if word in W2v.wv:
            vectors.append(W2v.wv[word])
    if len(vectors) == 0:
        return np.zeros(dimension)    
    else:
        return np.mean(vectors,axis = 0).astype(np.float32)



#intereface
st.title(":reb[Fake] news Detector:detector:")


st.subheader("Compare Logistic Regression vs. Neural Networks")


user_input = st.text_area("Paste the news article text here:", height=200)

if st.button("Analyze News"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
         clened_text = clear(user_input)
         tf = tfidf.transform([clened_text])
         lr_pd = lr.predict(tf)[0]
         lr_prob = lr.predict_proba(tf)[0][1]


         w2v_embedded_text = vecsent(clened_text.split())
         input_tensor = torch.from_numpy(w2v_embedded_text).unsqueeze(0)

         with torch.no_grad():
              raw_out = model(input_tensor)
              nn_prod = torch.sigmoid(raw_out).item()
              nn_pricd = 1 if nn_prod >0.5 else 0
        
        
         col1 , col2 = st.columns(2)

        
        
         with col1:
            st.write("### Logistic Regression")
            label = "🚨 FAKE" if lr_pd == 1 else "✅ REAL"
            st.metric("Result", label)
            st.write(f"Confidence: {lr_prob:.2%}")

         with col2:
            st.write("### Neural Network")
            label = "🚨 FAKE" if nn_pricd == 1 else "✅ REAL"
            st.metric("Result", label)
            st.write(f"Confidence: {nn_prod:.2%}")
        
         st.divider()
         st.write("### 📝 Article Summary")
         with st.spinner("Generating summary..."):
            
            if len(user_input.split()) > 50:
                summary_output = summarizer(user_input, max_length=150, min_length=50, do_sample=False)
                st.info(summary_output[0]['summary_text'])
            else:
                st.write("Text is too short to summarize.")