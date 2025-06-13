"""
Penguin Species Classification Model Training Script

This script trains a Random Forest classifier to predict penguin species
based on physical characteristics from the Palmer Penguins dataset.

Author: Your Name
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

def load_and_clean_data(file_path='data/penguins.csv'):
    """
    Load and clean the penguin dataset
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned penguin data
    """
    try:
        penguin_df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {len(penguin_df)} records")
        
        # Display basic info about the dataset
        print(f"Missing values before cleaning: {penguin_df.isnull().sum().sum()}")
        
        # Remove rows with missing values
        penguin_df.dropna(inplace=True)
        print(f"Records after cleaning: {len(penguin_df)}")
        
        return penguin_df
        
    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}")
        return None

def prepare_features_and_target(penguin_df):
    """
    Prepare features and target variables for machine learning
    
    Args:
        penguin_df (pd.DataFrame): Cleaned penguin data
        
    Returns:
        tuple: (features, target, target_mapping)
    """
    # Define target variable (species)
    target = penguin_df['species']
    
    # Define feature variables
    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 
                          'flipper_length_mm', 'body_mass_g', 'sex']]
    
    # Convert categorical variables to dummy variables
    features = pd.get_dummies(features)
    
    # Encode target variable to numerical format
    target_encoded, target_mapping = pd.factorize(target)
    
    print(f"Features shape: {features.shape}")
    print(f"Target classes: {target_mapping}")
    
    return features, target_encoded, target_mapping

def train_model(features, target, test_size=0.2, random_state=15):
    """
    Train a Random Forest classifier
    
    Args:
        features (pd.DataFrame): Feature variables
        target (np.array): Target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (trained_model, test_accuracy, feature_names)
    """
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    
    # Initialize and train Random Forest classifier
    rfc = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        max_depth=10
    )
    rfc.fit(x_train, y_train)
    
    # Make predictions on test set
    y_pred = rfc.predict(x_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    
    return rfc, accuracy, features.columns

def save_model_and_mappings(model, target_mapping, model_dir='models'):
    """
    Save the trained model and target mappings to pickle files
    
    Args:
        model: Trained Random Forest model
        target_mapping: Species name mappings
        model_dir (str): Directory to save model files
    """
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the trained model
    model_path = os.path.join(model_dir, 'random_forest_penguin.pickle')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # Save the target mappings
    mapping_path = os.path.join(model_dir, 'output_penguin.pickle')
    with open(mapping_path, 'wb') as f:
        pickle.dump(target_mapping, f)
    print(f"Target mappings saved to: {mapping_path}")

def create_feature_importance_plot(model, feature_names, save_path='images/feature_importance.png'):
    """
    Create and save a feature importance plot
    
    Args:
        model: Trained Random Forest model
        feature_names: Names of the features
        save_path (str): Path to save the plot
    """
    # Create images directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create horizontal bar plot
    sns.barplot(x=importances, y=feature_names, orient='h')
    plt.title('Feature Importance for Penguin Species Classification', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to: {save_path}")

def main():
    """
    Main function to execute the complete training pipeline
    """
    print("=" * 60)
    print("PENGUIN SPECIES CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load and clean data
    penguin_df = load_and_clean_data()
    if penguin_df is None:
        return
    
    # Step 2: Prepare features and target
    features, target, target_mapping = prepare_features_and_target(penguin_df)
    
    # Step 3: Train the model
    model, accuracy, feature_names = train_model(features, target)
    
    # Step 4: Save model and mappings
    save_model_and_mappings(model, target_mapping)
    
    # Step 5: Create feature importance plot
    create_feature_importance_plot(model, feature_names)
    
    print("=" * 60)
    print(f"MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()