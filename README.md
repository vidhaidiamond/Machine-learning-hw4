# Machine-learning-hw4
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Download needed resources (only needed once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# --- Helper: Convert NLTK POS → WordNet POS ---
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# --- Main Function ---
def process_text(text):
    # 1. Tokenization
    tokens = word_tokenize(text)

    # 2. Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w.lower() not in stop_words and w.isalpha()]

    # 3. POS tagging
    pos_tags = nltk.pos_tag(filtered)

    # 4. Lemmatize & keep only nouns and verbs
    lemmatizer = WordNetLemmatizer()
    final_words = []

    for word, tag in pos_tags:
        wn_tag = get_wordnet_pos(tag)
        
        # Only nouns (NN*) or verbs (VB*)
        if tag.startswith('N') or tag.startswith('V'):
            if wn_tag:
                lemma = lemmatizer.lemmatize(word, wn_tag)
            else:
                lemma = lemmatizer.lemmatize(word)
            final_words.append(lemma)

    return final_words


# --- Example ---
text = "The children are playing outside while the dog runs in the garden."

print(process_text(text))
