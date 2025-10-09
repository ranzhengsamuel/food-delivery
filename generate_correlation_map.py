
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

def generate_correlation_map():
    """
    Loads the food delivery dataset, calculates the correlation matrix,
    and saves a heatmap visualization as a PNG file.
    """
    try:
        # --- Data Loading ---
        # Get the absolute path of the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full path to the data file
        data_path = os.path.join(script_dir, "data", "Food_Delivery_Times.csv")
        
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)

        # --- Data Preprocessing for Correlation ---
        # Create a copy to avoid modifying the original dataframe
        corr_data = data.copy()

        # Use LabelEncoder to convert categorical columns to numerical format
        le = LabelEncoder()
        for col in corr_data.columns:
            if corr_data[col].dtype == 'object':
                corr_data[col] = le.fit_transform(corr_data[col])

        # --- Correlation Matrix Calculation ---
        print("Calculating correlation matrix...")
        corr_matrix = corr_data.corr()

        # --- Heatmap Generation ---
        print("Generating heatmap...")
        plt.figure(figsize=(12, 10))  # Set the figure size for better readability
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis', linewidths=.5)
        plt.title('Correlation Heatmap of Food Delivery Features', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()  # Adjust layout to make sure everything fits

        # --- Save the Figure ---
        output_path = os.path.join(script_dir, "correlation_heatmap.png")
        plt.savefig(output_path)
        print(f"Successfully saved correlation heatmap to: {output_path}")

    except FileNotFoundError:
        print(f"Error: The file was not found at {data_path}. Please ensure the 'data' folder and 'Food_Delivery_Times.csv' are correctly placed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_correlation_map()
