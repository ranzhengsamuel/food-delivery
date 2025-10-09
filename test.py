
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import os

# Set page title
st.set_page_config(page_title="Correlation Heatmap")

# Title for the app
st.title("Correlation Heatmap for Food Delivery Data")

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads the dataset from a local CSV file using a relative path."""
    try:
        # Get the absolute path of the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full path to the data file
        data_path = os.path.join(script_dir, "data", "Food_Delivery_Times.csv")
        # Read the CSV file
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file was not found at {data_path}. Please make sure the 'data' folder and 'Food_Delivery_Times.csv' exist in the same directory as the script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# Load the data
data = load_data()

# --- Correlation Heatmap ---
if data is not None:
    st.header("Correlation Heatmap of Features")

    # Create a copy of the data to avoid modifying the original dataframe
    corr_data = data.copy()

    # Use LabelEncoder to convert categorical columns to numerical format
    # A correlation matrix can only be calculated on numerical data.
    le = LabelEncoder()
    for col in corr_data.columns:
        if corr_data[col].dtype == 'object':
            corr_data[col] = le.fit_transform(corr_data[col])

    # Calculate the correlation matrix
    corr_matrix = corr_data.corr()

    # Create the heatmap figure using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',  # You can choose other colorscales like 'Plasma', 'Blues', etc.
        hoverongaps=False
    ))

    # Update layout for better readability
    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis_tickangle=-45
    )

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Data could not be loaded. The correlation heatmap cannot be displayed.")
