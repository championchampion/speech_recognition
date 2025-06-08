import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import streamlit as st
import speech_recognition as sr
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Chatbot avec saisie texte et vocale")
st.write("Chargement des données...")

with open('chatbot_data.txt', 'r', encoding='utf8') as file:
    raw_text = file.read().lower()
st.write("Données chargées.")

sentence_tokens = sent_tokenize(raw_text)
#sentence_tokens = sent_tokenize(raw_text, language='english')

st.write(f"{len(sentence_tokens)} phrases chargées.")

def chatbot_response(user_input):
    sentence_tokens.append(user_input)
    vectorizer = TfidfVectorizer().fit_transform(sentence_tokens)
    vectors = vectorizer.toarray()
    cosine_vals = cosine_similarity(vectors[-1].reshape(1, -1), vectors[:-1])
    idx = cosine_vals.argsort()[0][-1]

    sentence_tokens.pop()

    flat = cosine_vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]

    if req_tfidf == 0:
        return "Je suis désolé, je ne comprends pas votre question."
    else:
        return sentence_tokens[idx]

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Parlez maintenant...")
        audio = recognizer.listen(source, timeout=5)
    try:
        text = recognizer.recognize_google(audio, language='fr-FR')
        st.write(f"Vous avez dit : {text}")
        return text
    except Exception:
        st.write("Je n'ai pas pu comprendre l'audio. Veuillez réessayer.")
        return ""

mode = st.radio("Choisissez le mode d'entrée :", ("Texte", "Parole"))
st.write(f"Mode choisi : {mode}")

if mode == "Texte":
    user_input = st.text_input("Tapez votre message ici :")
    if st.button("Envoyer"):
        if user_input:
            response = chatbot_response(user_input.lower())
            st.write("Chatbot :", response)

elif mode == "Parole":
    if st.button("Parler"):
        user_input = speech_to_text()
        if user_input:
            response = chatbot_response(user_input.lower())
            st.write("Chatbot :", response)
