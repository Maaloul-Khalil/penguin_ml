"""
Penguin Species Classifier - Streamlit Web Application

This app predicts penguin species based on physical characteristics
using a Random Forest machine learning model trained on Palmer Penguins dataset.

Author: Your Name
Date: 2025
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="🐧 Penguin Species Classifier",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_default_models():
    """
    Load pre-trained model and mappings from pickle files
    
    Returns:
        tuple: (model, species_mapping) or (None, None) if files don't exist
    """
    model_path = 'models/random_forest_penguin.pickle'
    mapping_path = 'models/output_penguin.pickle'
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(mapping_path, 'rb') as f:
            species_mapping = pickle.load(f)
        return model, species_mapping
    except FileNotFoundError:
        return None, None

def train_new_model(penguin_df):
    """
    Train a new Random Forest model on uploaded data
    
    Args:
        penguin_df (pd.DataFrame): Penguin dataset
        
    Returns:
        tuple: (model, species_mapping, accuracy)
    """
    # Clean the data
    penguin_df = penguin_df.dropna()
    
    # Prepare features and target
    target = penguin_df['species']
    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
                          'flipper_length_mm', 'body_mass_g', 'sex']]
    
    # Convert categorical variables to dummy variables
    features = pd.get_dummies(features)
    
    # Encode target variable
    target_encoded, species_mapping = pd.factorize(target)
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        features, target_encoded, test_size=0.2, random_state=15
    )
    
    # Train the model
    model = RandomForestClassifier(random_state=15, n_estimators=100)
    model.fit(x_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, species_mapping, accuracy, features.columns

def create_species_distribution_plots(penguin_df):
    """
    Create distribution plots for key features by species
    
    Args:
        penguin_df (pd.DataFrame): Penguin dataset
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bill Length Distribution by Species")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=penguin_df, x='bill_length_mm', hue='species', 
                    multiple="overlay", alpha=0.7, ax=ax)
        plt.title('Bill Length by Species')
        plt.xlabel('Bill Length (mm)')
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Flipper Length Distribution by Species")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=penguin_df, x='flipper_length_mm', hue='species', 
                    multiple="overlay", alpha=0.7, ax=ax)
        plt.title('Flipper Length by Species')
        plt.xlabel('Flipper Length (mm)')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Bill Depth Distribution by Species")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=penguin_df, x='bill_depth_mm', hue='species', 
                    multiple="overlay", alpha=0.7, ax=ax)
        plt.title('Bill Depth by Species')
        plt.xlabel('Bill Depth (mm)')
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Body Mass Distribution by Species")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=penguin_df, x='body_mass_g', hue='species', 
                    multiple="overlay", alpha=0.7, ax=ax)
        plt.title('Body Mass by Species')
        plt.xlabel('Body Mass (g)')
        st.pyplot(fig)
        plt.close()

def display_feature_importance():
    """Display feature importance plot if it exists"""
    importance_path = 'images/feature_importance.png'
    if os.path.exists(importance_path):
        st.subheader("Feature Importance")
        st.image(importance_path, caption="Features ranked by importance for species prediction")
    else:
        st.info("Feature importance plot not available. Run the training script to generate it.")

def get_user_input():
    """
    Create input form for penguin characteristics
    
    Returns:
        list: User input values in the correct order for prediction
    """
    st.sidebar.header("🐧 Penguin Characteristics")
    st.sidebar.write("Enter the penguin's measurements:")
    
    # Input fields
    island = st.sidebar.selectbox(
        'Island', 
        options=['Biscoe', 'Dream', 'Torgersen'],
        help="Island where the penguin was observed"
    )
    
    sex = st.sidebar.selectbox(
        'Sex', 
        options=['Female', 'Male'],
        help="Penguin's sex"
    )
    
    bill_length = st.sidebar.number_input(
        'Bill Length (mm)', 
        min_value=0.0, 
        max_value=100.0, 
        value=45.0,
        step=0.1,
        help="Length of the penguin's bill in millimeters"
    )
    
    bill_depth = st.sidebar.number_input(
        'Bill Depth (mm)', 
        min_value=0.0, 
        max_value=50.0, 
        value=17.0,
        step=0.1,
        help="Depth of the penguin's bill in millimeters"
    )
    
    flipper_length = st.sidebar.number_input(
        'Flipper Length (mm)', 
        min_value=0.0, 
        max_value=300.0, 
        value=200.0,
        step=1.0,
        help="Length of the penguin's flipper in millimeters"
    )
    
    body_mass = st.sidebar.number_input(
        'Body Mass (g)', 
        min_value=0.0, 
        max_value=10000.0, 
        value=4000.0,
        step=50.0,
        help="Body mass of the penguin in grams"
    )
    
    # Convert categorical inputs to dummy variables
    island_biscoe = 1 if island == 'Biscoe' else 0
    island_dream = 1 if island == 'Dream' else 0
    island_torgersen = 1 if island == 'Torgersen' else 0
    
    sex_female = 1 if sex == 'Female' else 0
    sex_male = 1 if sex == 'Male' else 0
    
    # Return features in the correct order for prediction
    return [bill_length, bill_depth, flipper_length, body_mass, 
            island_biscoe, island_dream, island_torgersen, sex_female, sex_male]

def main():
    """Main application function"""
    
    # App title and description
    st.title("🐧 Penguin Species Classifier")
    st.write("""
    This app uses machine learning to predict penguin species based on physical characteristics.
    The model is trained on the famous Palmer Penguins dataset and uses a Random Forest classifier.
    
    **How to use:**
    1. Upload your own penguin data (optional) or use the pre-trained model
    2. Enter penguin measurements in the sidebar
    3. Get instant species prediction!
    """)
    
    # Model and data initialization
    model = None
    species_mapping = None
    penguin_df = None
    
    # File upload section
    st.subheader("📁 Data Upload (Optional)")
    uploaded_file = st.file_uploader(
        "Upload your own penguin dataset (CSV format)",
        type=['csv'],
        help="Upload a CSV file with penguin data to train a new model"
    )
    
    if uploaded_file is not None:
        try:
            # Load uploaded data
            penguin_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Data uploaded successfully! ({len(penguin_df)} records)")
            
            # Display data preview
            with st.expander("📊 Data Preview"):
                st.dataframe(penguin_df.head())
                st.write(f"Dataset shape: {penguin_df.shape}")
                st.write("Missing values:", penguin_df.isnull().sum().to_dict())
            
            # Train new model
            with st.spinner("🤖 Training new model..."):
                model, species_mapping, accuracy, feature_names = train_new_model(penguin_df)
            
            st.success(f"🎯 Model trained successfully! Accuracy: {accuracy:.2%}")
            
            # Display distribution plots
            if st.checkbox("📈 Show Species Distribution Plots"):
                create_species_distribution_plots(penguin_df)
                
        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {str(e)}")
            
    else:
        # Load default pre-trained model
        model, species_mapping = load_default_models()
        
        if model is None:
            st.warning("""
            ⚠️ Pre-trained model not found. Please either:
            1. Upload your own penguin dataset above, or
            2. Run the training script to generate the model files
            """)
            st.stop()
        else:
            st.success("✅ Using pre-trained model")
            
            # Display feature importance
            display_feature_importance()
    
    # Prediction section
    if model is not None and species_mapping is not None:
        st.subheader("🔮 Make a Prediction")
        
        # Get user input
        user_input = get_user_input()
        
        # Make prediction button
        if st.sidebar.button("🚀 Predict Species", type="primary"):
            try:
                # Make prediction
                prediction = model.predict([user_input])
                predicted_species = species_mapping[prediction[0]]
                
                # Get prediction probability
                prediction_proba = model.predict_proba([user_input])
                confidence = max(prediction_proba[0]) * 100
                
                # Display results
                st.success(f"""
                ## 🐧 Prediction Result
                
                **Predicted Species: {predicted_species}**
                
                **Confidence: {confidence:.1f}%**
                """)
                
                # Display prediction probabilities for all species
                st.subheader("📊 Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Species': species_mapping,
                    'Probability': prediction_proba[0]
                }).sort_values('Probability', ascending=False)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=prob_df, x='Probability', y='Species', ax=ax)
                plt.title('Species Prediction Probabilities')
                plt.xlabel('Probability')
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About the Palmer Penguins Dataset:**
    This dataset contains measurements of three penguin species (Adelie, Chinstrap, Gentoo) 
    observed on three islands (Biscoe, Dream, Torgersen) in the Palmer Archipelago, Antarctica.
    
    **Features used for prediction:**
    - Island location
    - Bill length and depth (mm)
    - Flipper length (mm)
    - Body mass (g)
    - Sex
    """)

if __name__ == "__main__":
    main()