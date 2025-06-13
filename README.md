# 🐧 Penguin Species Classifier

A machine learning web application that predicts penguin species based on physical characteristics using the Palmer Penguins dataset.

## Features
- Interactive web interface built with Streamlit
- Random Forest classifier for species prediction
- Feature importance visualization
- Option to upload custom penguin data
- Species distribution plots

## Setup Instructions

### Local Development
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the training script: `python src/train_model.py`
6. Run the Streamlit app: `streamlit run src/streamlit_app.py`

### Deployment on Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path to: `src/streamlit_app.py`
5. Deploy!

## Data Source
This project uses the Palmer Penguins dataset, which contains measurements for three penguin species observed on three islands in the Palmer Archipelago, Antarctica.

## Model Performance
The Random Forest classifier achieves high accuracy in predicting penguin species based on:
- Bill length and depth
- Flipper length
- Body mass
- Island location
- Sex# penguin_ml
