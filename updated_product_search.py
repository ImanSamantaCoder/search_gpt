import speech_recognition as sr
import streamlit as st
import pandas as pd
import json
import os
import ast
import requests
from io import BytesIO
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D, Input
from numpy.linalg import norm
import numpy as np

# Initialize NLTK and download 'punkt' tokenizer
nltk.download('punkt_tab')

# Sidebar Configuration
st.sidebar.title("🔍 Search Engine")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
df = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

user_query_input = st.sidebar.text_input(
    "Enter your product query", 
    "I need a black t-shirt for a festival under 600 rupees."
)
uploaded_image = st.sidebar.file_uploader(
    "Upload an image for image-based recommendations", 
    type=["jpg", "jpeg", "png"]
)

# Main Title
st.title("🛒 Comprehensive Search Engine")

@st.cache_data
def load_data(df):
    """
    Load and preprocess the dataset with pre-extracted features.
    """
    if df is None :
    # Load Dataset with Pre-Extracted Features
       df = pd.read_csv(r"D:\NLP_search_pravaah\full_product_features (1).csv")
    else :
       df = pd.read_csv(df)
    
    # Drop the 'brand' column if it exists
    if 'brand' in df.columns:
        df = df.drop('brand', axis=1)
    
    # Select and clean relevant columns
    combined_df = df[['name', 'additionalInfoTitle1', 'additionalInfoDescription1', 
                     'additionalInfoTitle2', 'additionalInfoDescription2', 'description']].fillna('')
    
    # Replace newline characters with spaces
    columns_to_clean = [
        'name',
        'additionalInfoTitle1',
        'additionalInfoDescription1',
        'additionalInfoTitle2',
        'additionalInfoDescription2',
        'description'
    ]
    
    for col in columns_to_clean:
        combined_df[col] = combined_df[col].str.replace('\n', ' ', regex=False).str.strip()
    
    # Merge columns into a single 'merged_column'
    df['merged_column'] = combined_df['name'] + ' ' + combined_df['additionalInfoTitle1'] + ' ' + \
                          combined_df['additionalInfoDescription1'] + ' ' + combined_df['additionalInfoTitle2'] + ' ' + \
                          combined_df['additionalInfoDescription2'] + ' ' + combined_df['description']
    
    # Limit the DataFrame to the first 144 entries if necessary
    df = df[:144]
    
    # Initialize NLTK's SnowballStemmer
    stemmer = SnowballStemmer("english")
    
    def tokenize_stem(text):
        tokens = word_tokenize(text.lower())  # Correctly using 'punkt'
        stemmed = [stemmer.stem(w) for w in tokens]
        return " ".join(stemmed)
    
    # Apply tokenization and stemming
    df['stemmed_tokens'] = df['merged_column'].apply(tokenize_stem)
    
    return df

df = load_data(df)

@st.cache_data
def load_feature_data():
    """
    Load the DataFrame with pre-extracted image features.
    """
    # Load the DataFrame with Pre-Extracted Features
    data = pd.read_csv(r'full_product_features (1).csv')  # Update with your actual path
    
    # Ensure 'feature_list' is in list format
    data['feature_list'] = data['feature_list'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Convert 'feature_list' column to NumPy array for efficient computation
    feature_list = np.array(data['feature_list'].tolist())
    
    return data, feature_list

data, feature_list = load_feature_data()

def initialize_models(groq_api_key):
    """
    Initialize the ChatGroq model with the provided API key.
    """
    # Set the environment variable for the API key
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    # Initialize the ChatGroq model
    model = ChatGroq(model="llama3-70b-8192")
    return model

if groq_api_key:
    chatgroq_model = initialize_models(groq_api_key)
else:
    st.sidebar.warning("⚠️ Please enter your Groq API Key to use the recommendation system.")
    st.stop()

def extract_product_details(user_query, model):
    """
    Use ChatGroq to extract product details from the user's query.
    """
    # Define the prompt template
    template = """
    You are a helpful assistant specializing in extracting product details. 
    From the User Input, extract the following information: productname(name of product),brand,gender (men/women),color,occasion,specification(ram,rom,size),price
    give product name carefully
    
    Please return the extracted information in JSON format without any additional text.
    
    User Input: "{input_text}"
    """
    
    # Create the ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_template(template)
    
    # Format the messages using the template
    messages = chat_prompt.format_messages(
        input_text=user_query
    )
    
    # Invoke the model with the formatted messages
    response = model.invoke(messages)
    
    try:
        json_data = json.loads(response.content)
        st.write(json_data)
        return json_data
    except json.JSONDecodeError:
        st.error("❌ Could not decode JSON from the model's response.")
        return {}

def tokenize_stem(text):
    """
    Tokenize and stem the input text.
    """
    stemmer = SnowballStemmer("english")
    tokens = word_tokenize(text.lower())  # Correctly using 'punkt'
    stemmed = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)

def get_recommendations(user_input, recommendations_df, similarity_threshold=0.30):
    """
    Generate recommendations based on cosine similarity of embeddings.
    """
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Compute embeddings for product descriptions
    product_embeddings = embedding_model.encode(recommendations_df['stemmed_tokens'].tolist())
    
    # Encode user input
    user_embedding = embedding_model.encode([user_input])
    
    # Calculate cosine similarity between user input and product embeddings
    sim_scores = cosine_similarity(user_embedding, product_embeddings).flatten()
    
    # Filter products with similarity greater than the threshold
    top_indices = [i for i, score in enumerate(sim_scores) if score > similarity_threshold]
    
    # Return the filtered recommendations based on similarity
    filtered_recommendations_df = recommendations_df.iloc[top_indices].copy()
    filtered_recommendations_df['similarity'] = sim_scores[top_indices]
    
    return filtered_recommendations_df

def check_query_match(specification, product_description, model):
    """
    Use ChatGroq to determine if the product matches the user's specification.
    """
    prompt = f"""
    User Query: "{specification}"

    Product Description: "{product_description}"

    Determine if the product matches the user's query. Answer with "Yes" or "No" without any explanation.
    Treat similar items intelligently, e.g., if a user wants a 6 GB RAM mobile, recommend an 8 GB RAM mobile if it's within limits.
    Do not confuse categories like smart watches with smartphones.
    """
    response = model.invoke(prompt)
    response_text = response.content.strip().lower()
    return response_text

def extract_price_constraints(user_query, model):
    """
    Extract price constraints from the user's query using ChatGroq.
    """
    prompt = f"""
    Extract any price constraints from the following user query.
    If the query mentions a price limit, specify if it is 'under' or 'above' carefully, and the numerical value.
    Return the result in JSON format as:
    {{
        "condition": "under" or "above",
        "price": numerical value or None
    }}
    Do not provide any additional text.
    User Query: "{user_query}"
    """
    response = model.invoke(prompt)
    try:
        constraints = json.loads(response.content)
        st.write(constraints)
        return constraints
    except json.JSONDecodeError:
        return {"condition": None, "price": None}

def filter_by_query_match(specification, filtered_recommendations_df, model):
    """
    Further filter recommendations based on query matching.
    """
    final_recommendations = []
    
    # Loop through the filtered recommendations and apply query matching
    for idx, row in filtered_recommendations_df.iterrows():
        product_description = row['stemmed_tokens']
        
        # Use ChatGroq to check if the user query matches the product description
        match_response = check_query_match(specification, product_description, model)
        
        # If the response is "Yes", keep the product
        if match_response == "yes":
            final_recommendations.append(row)
    
    # Return the final filtered recommendations as a DataFrame
    if final_recommendations:
        return pd.DataFrame(final_recommendations)
    else:
        return pd.DataFrame(columns=filtered_recommendations_df.columns)

def filter_by_price(recommendations_df, user_query, model):
    """
    Filter recommendations based on extracted price constraints.
    """
    constraints = extract_price_constraints(user_query, model)
    price_condition = constraints.get("condition")
    price_value = constraints.get("price")
    
    if price_value is not None:
        if price_condition == 'under':
            return recommendations_df[recommendations_df['price'] < price_value]
        elif price_condition == 'above':
            return recommendations_df[recommendations_df['price'] > price_value]
    return recommendations_df  # No price filter if not found

def perform_recommendation(user_query, model, df):
    """
    Orchestrate the recommendation process.
    """
    # Extract product details
    extracted_attributes = extract_product_details(user_query, model)
    st.write("### Extracted Attributes:", extracted_attributes)
    
    # Extract individual attributes with fallback
    name = str(extracted_attributes.get('productname', ''))
    brand = str(extracted_attributes.get('brand', ''))
    color = str(extracted_attributes.get('color', ''))
    price = extracted_attributes.get('price', None)
    gender = str(extracted_attributes.get('gender', ''))
    specification = str(extracted_attributes.get('specification', ''))
    occasion = str(extracted_attributes.get('occasion', ''))
    
    # Prepare user input for recommendations
    user_input_product = tokenize_stem(f"{gender} {color} {name} {brand}".strip())
    
    # Tokenize user query
    user_query_tokenized = tokenize_stem(user_query.strip())
    
    # Step 1: Cosine similarity filtering
    cosine_filtered_recommendations = get_recommendations(user_input_product, df, similarity_threshold=0.30)
    
    # Step 2: Query matching with ChatGroq
    if specification:
        specification_recommendations = filter_by_query_match(specification, cosine_filtered_recommendations, model)
    else:
        specification_recommendations = cosine_filtered_recommendations
    
    # Fallback if recommendations are too few
    if len(specification_recommendations) < 2:
        specification_recommendations = cosine_filtered_recommendations
    
    # Step 3: Price-based filtering
    final_recommendations = filter_by_price(specification_recommendations, user_query_tokenized, model)
    
    # Fallback if no recommendations after price filtering
    if final_recommendations.empty:
        final_recommendations = specification_recommendations
    
    return final_recommendations
if st.sidebar.button("🚀 Get Recommendations"):
    with st.spinner("🔍 Processing your request..."):
        final_recommendations = perform_recommendation(user_query_input, chatgroq_model, df)
        st.success("✅ Recommendations Generated!")
        st.dataframe(final_recommendations)
# Display DataFrame Head
st.header("📊 Dataset Preview")
st.dataframe(df.head())


# Image-Based Recommendation Section
st.header("📸 Image-Based Recommendations")

def extract_features_from_uploaded_image(uploaded_image):
    """
    Extract feature vector from the uploaded image using ResNet50.
    """
    img = Image.open(uploaded_image)
    
    # Convert image to RGB if it has an alpha channel
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    # Load ResNet50 model with GlobalMaxPooling2D
    resnet_model = ResNet50(weights='imagenet', include_top=False)
    resnet_model = tf.keras.Sequential([
        resnet_model,
        GlobalMaxPooling2D()
    ])
    
    # Extract features
    img_features = resnet_model.predict(preprocessed_img).flatten()
    normalized_features = img_features / norm(img_features)
    
    return normalized_features


if uploaded_image is not None:
    st.image(uploaded_image, caption='📷 Uploaded Image.', use_column_width=True)
    
    # Extract features from the uploaded image
    img_features = extract_features_from_uploaded_image(uploaded_image)
    
    # Perform Nearest Neighbors Search
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([img_features])
    
    st.subheader("🔍 Similar Products:")
    similar_products = data.iloc[indices[0]]
    st.write(similar_products)
else:
    st.write("🖼️ Please upload an image to get image-based recommendations.")
import streamlit as st
import sounddevice as sd
import numpy as np
import soundfile as sf
import io
import speech_recognition as sr

# Set audio parameters
samplerate = 44100  # Sample rate
duration = 10  # seconds

def record_audio():
    st.write("Recording... Please speak.")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    st.write("Recording finished.")
    return audio_data

def audiorec_demo_app(df,model):
    if st.sidebar.button("Start Recording"):
        audio_data = record_audio()
        
        # Save audio data to a WAV file in memory
        wav_file = io.BytesIO()
        sf.write(wav_file, audio_data, samplerate, format='WAV')
        wav_file.seek(0)

        # Playback
        st.audio(wav_file, format='audio/wav')

        # Transcribe audio
        user_query_input = transcribe_audio(wav_file.getvalue())
        final_recommendations = perform_recommendation(user_query_input, chatgroq_model, df)
        st.write(final_recommendations)
        
        if user_query_input :
            st.write("Transcribed Text:")
            st.write(user_query_input)
        else:
            st.error("Could not transcribe audio.")

def transcribe_audio(audio_data):
    recognizer = sr.Recognizer()
    audio_file = io.BytesIO(audio_data)

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    try:
        user_query_input = recognizer.recognize_google(audio)
        return user_query_input
    except sr.UnknownValueError:
        st.error("Speech could not be understood.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None

audiorec_demo_app(df,chatgroq_model)
    