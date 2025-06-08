import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # Télécharger la bonne ressource

texte = "Bonjour. Ceci est un test. Est-ce que ça fonctionne ?"
phrases = sent_tokenize(texte)
print(phrases)

