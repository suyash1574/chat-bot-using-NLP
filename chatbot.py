import nltk
import random
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load the corpus
try:
    with open('chatbot_corpus.txt', 'r', errors='ignore') as file:
        raw_text = file.read().lower()
except FileNotFoundError:
    raise FileNotFoundError("chatbot_corpus.txt is missing. Create the file in the same directory.")

# Tokenize sentences and words
sentence_tokens = nltk.sent_tokenize(raw_text)
word_tokens = nltk.word_tokenize(raw_text)

# Lemmatizer setup
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    """Lemmatizes words."""
    return [lemmatizer.lemmatize(token) for token in tokens]

def normalize_text(text):
    """Cleans and normalizes input text."""
    return lemmatize_tokens(
        nltk.word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation)))
    )

# Greeting responses
GREETING_INPUTS = ("hello", "hi", "greetings", "hey")
GREETING_RESPONSES = ["Hi there!", "Hello!", "Hey!", "Hi! How can I assist you?"]

def greet(sentence):
    """Checks for greeting and returns appropriate response."""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None


# Generate response using cosine similarity
def generate_response(user_input):
    """Generates a response based on input similarity to the corpus."""
    # Normalize user input
    user_input_tokens = normalize_text(user_input)
    sentence_tokens.append(user_input)  # Temporarily add user input to the token list

    # Vectorize sentences
    vectorizer = CountVectorizer().fit_transform(sentence_tokens)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(vectors[-1].reshape(1, -1), vectors[:-1])
    
    # Get index of most similar sentence
    index = cosine_sim.argmax()  # Index of the sentence with the highest match

    # If no match, send a fallback response
    if cosine_sim[0][index] < 0.1:  # Adjust threshold if necessary
        response = "I am sorry, I don't understand that."
    else:
        response = sentence_tokens[index]

    # Remove user input from tokens
    sentence_tokens.pop(-1)
    return response


# Main chatbot function
def chatbot():
    """Main chatbot loop."""
    print("Chatbot: Hello! I am a chatbot. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("Chatbot: Goodbye! Have a great day!")
            break
        greeting_response = greet(user_input)
        if greeting_response:
            print(f"Chatbot: {greeting_response}")
        else:
            print(f"Chatbot: {generate_response(user_input)}")


# Run the chatbot
if __name__ == "__main__":
    chatbot()
