import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import requests
from PIL import Image
from io import BytesIO
import os
import gc

# Set page config
st.set_page_config(page_title="Recipe Generator", page_icon="🍲", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; padding: 20px; }
    .stButton>button { background-color: #ff6347; color: white; border-radius: 8px; font-weight: bold; }
    .stButton>button:hover { background-color: #e5533d; }
    .sidebar .sidebar-content { background-color: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .recipe-card { background-color: #ffffff; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .recipe-title { font-size: 24px; color: #333; font-weight: bold; margin-bottom: 10px; }
    .recipe-details { font-size: 16px; color: #555; }
    .recipe-image { border-radius: 10px; max-width: 100%; }
    h1, h2 { color: #333; font-family: 'Arial', sans-serif; }
    .stTextArea textarea { border-radius: 5px; }
    .stSpinner { margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# Data processing functions
def extract_ingredient_names(ingredients):
    if not isinstance(ingredients, list):
        return []
    ingredient_names = []
    for ingredient in ingredients:
        parts = ingredient.lower().split()
        # Remove quantities, numbers, and descriptors
        parts = [part for part in parts if not any(c.isdigit() for c in part) and part not in ('ml', 'g', 'tbsp', 'tsp', 'large', 'pack')]
        ingredient_names.append(parts[-1] if parts else "")
    return [name for name in ingredient_names if name]

def encode_ingredients(recipe):
    encoding = np.zeros(len(unique_ingredients))
    for ingredient in recipe:
        if ingredient in ingredient_to_index:
            encoding[ingredient_to_index[ingredient]] = 1
    return encoding

# Load and preprocess data
try:
    df = pd.read_json('bbcgoodfood_recipes2.json')
    ddf = df.transpose()
except FileNotFoundError:
    st.error("Dataset 'bbcgoodfood_recipes2.json' not found. Please upload the file.")
    st.stop()

# Normalize data
ddf['Ingredient Names'] = ddf['Ingredients'].apply(extract_ingredient_names)
unique_ingredients = set([ingredient for ingredients_list in ddf['Ingredient Names'] for ingredient in ingredients_list if ingredient])
ingredient_to_index = {ingredient: i for i, ingredient in enumerate(unique_ingredients)}
ddf['Encoded Ingredients'] = ddf['Ingredient Names'].apply(encode_ingredients)
X_train = np.array(list(ddf['Encoded Ingredients']))

# Load or train model (simplified, but optional for filtering)
MODEL_PATH = "recipe_model.h5"

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            st.success("Loaded pre-trained model!")
            return model
        except Exception as e:
            st.warning(f"Failed to load model: {e}. Training new model...")
    
    # Train new model (minimal for Render)
    model = Sequential([
        Dense(16, activation='relu', input_shape=(len(unique_ingredients),)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')  # Simplified output
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    subset_size = min(200, len(X_train))
    # Dummy labels for simplicity (not used in filtering)
    y_train = np.random.randint(0, 2, size=(subset_size,))
    model.fit(X_train[:subset_size], y_train, epochs=3, batch_size=64, verbose=1)
    try:
        model.save(MODEL_PATH)
        st.success("Trained and saved new model!")
    except Exception as e:
        st.warning(f"Failed to save model: {e}")
    return model

model = get_model()

# Recipe recommendation
def recommend_recipes(available_ingredients, ddf, difficulty_level):
    available_ingredients = [ing.lower().strip() for ing in available_ingredients if ing.strip()]
    if not available_ingredients:
        return pd.DataFrame()
    available_ingredients_set = set(available_ingredients)
    # Partial matching: at least one ingredient matches
    filtered_ddf = ddf[
        (ddf['Ingredient Names'].apply(lambda x: any(ing in available_ingredients_set for ing in x))) &
        (ddf['Difficulty'].str.lower() == difficulty_level.lower())
    ]
    if filtered_ddf.empty:
        return filtered_ddf
    # Sort by number of matching ingredients
    filtered_ddf = filtered_ddf.assign(
        MatchCount=ddf['Ingredient Names'].apply(lambda x: len(set(x) & available_ingredients_set))
    )
    filtered_ddf = filtered_ddf.sort_values(by='MatchCount', ascending=False)
    return filtered_ddf.head(3)

# Streamlit UI
st.title("🍴 Recipe Generator")
st.markdown("Discover delicious recipes based on your ingredients and skill level!")

# Sidebar
with st.sidebar:
    st.header("Recipe Settings")
    available_ingredients = st.text_area("Enter ingredients (comma-separated):", placeholder="sugar, butter, flour").split(',')
    difficulty_level = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
    generate_button = st.button("Generate Recipes")

# Main content
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Recommended Recipes")
    if generate_button:
        with st.spinner("Finding recipes..."):
            filtered_ddf = recommend_recipes(available_ingredients, ddf, difficulty_level)
            if not filtered_ddf.empty:
                for _, row in filtered_ddf.iterrows():
                    with st.container():
                        st.markdown(f"<div class='recipe-card'>", unsafe_allow_html=True)
                        st.markdown(f"<div class='recipe-title'>{row.get('Title', 'Untitled Recipe')}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='recipe-details'>**Difficulty**: {row.get('Difficulty', 'N/A')}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='recipe-details'>**Prep Time**: {row.get('Prep Time', 'N/A')}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='recipe-details'>**Cook Time**: {row.get('Cook Time', 'N/A')}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='recipe-details'>**Serves**: {row.get('Serves', 'N/A')}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='recipe-details'>**Ingredients**: {', '.join(row.get('Ingredients', []))}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='recipe-details'>**Instructions**: {', '.join(row.get('Method Steps', ['N/A']))}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("No recipes found. Try different ingredients or difficulty level.")
            gc.collect()

with col2:
    st.subheader("Recipe Image")
    if generate_button and not filtered_ddf.empty:
        top_recipe = filtered_ddf.iloc[0]
        image_url = top_recipe.get('Image', '')
        if image_url:
            try:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption=top_recipe.get('Title', 'Recipe Image'), use_column_width=True, clamp=True)
            except:
                st.info("Image not available.")
        else:
            st.info("No image provided for this recipe.")
