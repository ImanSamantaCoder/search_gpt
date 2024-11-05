import streamlit as st
import pandas as pd
import numpy as np
import ast
import speech_recognition as sr
import sounddevice as sd
import wavio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D, Input
from tensorflow.keras.models import Sequential
from numpy.linalg import norm

# Load the data
file_path = r"D:\NLP_search_pravaah\full_product_features.csv"
data = pd.read_csv(file_path)

# Fill missing values
data['product_name'].fillna('', inplace=True)
data['brand'].fillna('', inplace=True)
data['specification'].fillna('', inplace=True)
data['color'].fillna('', inplace=True)
data['type'].fillna('', inplace=True)
data['productdetails'].fillna('', inplace=True)
data['price'].fillna(0, inplace=True)

# Function to convert 'feature_list' to list
def convert_feature_list_to_list(feature_str):
    try:
        return ast.literal_eval(feature_str)  # Convert string to list
    except (ValueError, SyntaxError):
        return []  # Return an empty list if conversion fails

# Apply the conversion to 'feature_list'
data['feature_list'] = data['feature_list'].apply(convert_feature_list_to_list)

# Initialize the SentenceTransformer model
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function for text-based recommendation
def get_text_recommendations(user_input, num_recommendations=5):
    product_embeddings = sentence_model.encode(data['product_name'].tolist())
    user_embedding = sentence_model.encode([user_input])
    sim_scores = cosine_similarity(user_embedding, product_embeddings).flatten()
    top_indices = sim_scores.argsort()[-num_recommendations:][::-1]
    return data.iloc[top_indices][['product_name', 'brand', 'price', 'product_image_urls']]

# Load the ResNet50 model for image feature extraction
input_shape = (224, 224, 3)
image_model = Sequential(
    [
        Input(shape=input_shape),
        ResNet50(weights='imagenet', include_top=False),
        GlobalMaxPooling2D()
    ]
)
image_model.trainable = False

# Function to extract image features
def extract_image_features(img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = image_model.predict(preprocessed_img).flatten()
    return features / norm(features)

# Nearest neighbors search
feature_list_store = data['feature_list'].tolist()
feature_list = np.array(feature_list_store)
neighbours = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbours.fit(feature_list)

# Streamlit UI
st.title("NLP search engine 🛍️ 🛒  💳 ")

# User input for text-based search
user_input = st.text_input("Enter product description (e.g., 'men tshirt'): ")

# Upload image for image-based recommendation
uploaded_file = st.file_uploader("Or upload a product image:", type=["jpg", "png", "jpeg"])

# Audio recording parameters
fs = 44100  # Sample rate
duration = 5  # Duration of recording in seconds

# Button to start recording
if st.button("Record Audio"):
    st.write("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    st.write("Recording finished. Processing...")

    # Save the recorded audio to a WAV file
    wavio.write("recorded_audio.wav", recording, fs, sampwidth=2)

    # Use SpeechRecognition to convert audio to text
    recognizer = sr.Recognizer()
    with sr.AudioFile("recorded_audio.wav") as source:
        audio_content = recognizer.record(source)
        try:
            transcribed_text = recognizer.recognize_google(audio_content)
            st.write(f"Transcribed text: {transcribed_text}")
            recommendations_from_audio = get_text_recommendations(transcribed_text)
            st.subheader("Recommendations Based on Audio")
            for index, row in recommendations_from_audio.iterrows():
                st.write(f"**Product Name:** {row['product_name']}")
                st.write(f"**Brand:** {row['brand']}")
                st.write(f"**Price:** {row['price']}")
                
                # Directly display the image URL
                image_url = row['product_image_urls']  # Assuming it's a single URL
                st.image(image_url, use_column_width=True)  # Display the image
                
                st.write("---")  # Separator for each product
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")

# Display recommendations based on text input
if user_input:
    st.subheader("Recommendations Based on Text")
    recommendations = get_text_recommendations(user_input)

    # Display the recommendations with images
    for index, row in recommendations.iterrows():
        st.write(f"**Product Name:** {row['product_name']}")
        st.write(f"**Brand:** {row['brand']}")
        st.write(f"**Price:** {row['price']}")
        
        # Directly display the image URL
        image_url = row['product_image_urls']  # Assuming it's a single URL
        st.image(image_url, use_column_width=True)  # Display the image
        
        st.write("---")  # Separator for each product

# Display recommendations based on uploaded image
if uploaded_file is not None:
    st.subheader("Recommendations Based on Image")
    features = extract_image_features(uploaded_file)
    distances, indices = neighbours.kneighbors([features])
    image_recommendations = data.iloc[indices.flatten()][['product_name', 'brand', 'price', 'product_image_urls']]
    
    # Display the image recommendations
    for index, row in image_recommendations.iterrows():
        st.write(f"**Product Name:** {row['product_name']}")
        st.write(f"**Brand:** {row['brand']}")
        st.write(f"**Price:** {row['price']}")
        
        # Directly display the image URL
        image_url = row['product_image_urls']  # Assuming it's a single URL
        st.image(image_url, use_column_width=True)  # Display the image
        
        st.write("---")  # Separator for each product
