
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.impute import KNNImputer
import numpy as np

# Set page config
st.set_page_config(layout="wide", page_title="Food Delivery Analysis Dashboard")

# Title and description
st.title("Interactive Food Delivery Analysis")
st.write("An interactive dashboard for Exploratory Data Analysis (EDA) of food delivery datasets.")

# --- Data Loading ---
@st.cache_data
def load_data(file_name):
    """Loads a dataset from a local CSV file."""
    try:
        # Prioritize loading from a 'data' subdirectory if it exists
        data_path = os.path.join("data", file_name)
        if not os.path.exists(data_path):
            # Fallback to the root directory
            data_path = file_name
        
        df = pd.read_csv(data_path, na_values=['Not given'])
        return df
    except Exception as e:
        st.error(f"Error loading data from {file_name}: {e}")
        return None

# --- Sidebar for Dataset Selection ---
st.sidebar.header("Dataset Selection")
dataset_choice = st.sidebar.radio(
    "Choose a dataset to analyze:",
    ("Food Delivery Times", "NYC Food Orders")
)

# --- Main App Logic ---

if dataset_choice == "Food Delivery Times":
    st.header("Analysis of Food Delivery Times")
    data = load_data("Food_Delivery_Times.csv")

    if data is not None:
        # --- Sidebar Filters for Food Delivery Times ---
        st.sidebar.header("Filters")
        # Clean column names for display and use
        data.columns = [col.strip() for col in data.columns]
        
        # Check for required columns before creating filters
        required_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
        if all(col in data.columns for col in required_cols):
            # Remove NaN options from filters
            weather_options = data["Weather"].dropna().unique()
            traffic_options = data["Traffic_Level"].dropna().unique()
            
            selected_weather = st.sidebar.multiselect("Weather Conditions", weather_options, default=weather_options)
            selected_traffic = st.sidebar.multiselect("Road Traffic Density", traffic_options, default=traffic_options)

            # Filter data based on selections
            filtered_data = data[
                (data["Weather"].isin(selected_weather)) &
                (data["Traffic_Level"].isin(selected_traffic))
            ]
            
            st.subheader("Filtered Data Preview")
            st.dataframe(filtered_data.head())

            # --- Visualizations ---
            st.subheader("Data Visualizations")

            # 1. Distribution of Delivery Time
            st.write("### Distribution of Delivery Time")
            fig_delivery_time = px.histogram(filtered_data, x="Delivery_Time_min", nbins=30, title="Distribution of Delivery Time (min)")
            st.plotly_chart(fig_delivery_time, use_container_width=True)

            # 2. Correlation Heatmap
            st.write("### Correlation Heatmap of Numerical Features")
            # Select only numeric columns for correlation, excluding IDs
            numeric_cols = filtered_data.select_dtypes(include=['number']).drop(columns=['Order_ID'], errors='ignore')
            corr = numeric_cols.corr()
            fig_corr = px.imshow(corr, text_auto='.2f', aspect="auto", color_continuous_scale='Viridis', title="Correlation Heatmap of Numerical Features")
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # 3. Box Plots for Categorical Analysis
            st.write("### Categorical Feature Analysis")
            col1, col2 = st.columns(2)

            with col1:
                # Delivery Time by Weather Conditions
                fig_weather = px.box(filtered_data, x="Weather", y="Delivery_Time_min", title="Delivery Time by Weather Conditions")
                st.plotly_chart(fig_weather, use_container_width=True)
                
                # Delivery Time vs Time of Day
                fig_time_of_day = px.box(filtered_data, x="Time_of_Day", y="Delivery_Time_min", title="Delivery Time vs. Time of Day")
                st.plotly_chart(fig_time_of_day, use_container_width=True)

            with col2:
                # Delivery Time by Road Traffic Density
                fig_traffic = px.box(filtered_data, x="Traffic_Level", y="Delivery_Time_min", title="Delivery Time by Road Traffic Density")
                st.plotly_chart(fig_traffic, use_container_width=True)

                # Delivery Time vs Vehicle Type
                fig_vehicle_type = px.box(filtered_data, x="Vehicle_Type", y="Delivery_Time_min", title="Delivery Time vs. Vehicle Type")
                st.plotly_chart(fig_vehicle_type, use_container_width=True)

            # 4. Scatter Plots for Numerical Analysis
            st.write("### Numerical Feature Analysis")
            col3, col4 = st.columns(2)

            with col3:
                # Scatter plot of Courier Experience vs. Delivery Time
                fig_age_time = px.scatter(filtered_data, x="Courier_Experience_yrs", y="Delivery_Time_min",
                                          title="Courier Experience (yrs) vs. Delivery Time",
                                          trendline="ols")
                st.plotly_chart(fig_age_time, use_container_width=True)
            
            with col4:
                # Scatter plot of Distance vs. Delivery Time
                fig_ratings_time = px.scatter(filtered_data, x="Distance_km", y="Delivery_Time_min",
                                              title="Distance (km) vs. Delivery Time",
                                              trendline="ols")
                # Add annotation for the line equation
                fig_ratings_time.add_annotation(
                    x=0.05, y=0.95,
                    xref="paper", yref="paper",
                    text="y = 3x + 26.3",
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.5)"
                )
                st.plotly_chart(fig_ratings_time, use_container_width=True)

        else:
            st.warning("The 'Food Delivery Times' dataset is missing one or more required columns: 'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type'.")


elif dataset_choice == "NYC Food Orders":
    st.header("Analysis of NYC Food Orders")
    nyc_data = load_data("NYC_food_order.csv")

    if nyc_data is not None:
        # --- Data Overview ---
        st.subheader("Data Preview")
        st.dataframe(nyc_data.head())

        # --- Visualizations ---
        st.subheader("Data Visualizations")
        fig_delivery_dist = px.histogram(nyc_data, x="delivery_time", nbins=30, title="Distribution of Delivery Time (min)")
        st.plotly_chart(fig_delivery_dist, use_container_width=True)

        # Pre-process data, dropping ID columns for correlation
        nyc_numeric = nyc_data.select_dtypes(include=np.number).drop(columns=['order_id', 'customer_id'], errors='ignore')
        
        st.subheader("Correlation Heatmap Analysis")
        st.write("Exploring correlations within the NYC Food Order dataset based on different segmentations.")

        # --- Tabs for different views ---
        tab1, tab2, tab3, tab4 = st.tabs(["Overall Correlation", "Weekday vs. Weekend", "Cuisine Type", "Imputation Analysis"])

        with tab1:
            st.write("### Overall Numeric Correlation")
            corr_matrix = nyc_numeric.corr()
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu', title="Overall Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.write("### Correlation by Day of the Week")
            # Drop IDs before correlation
            weekday_df = nyc_data[nyc_data['day_of_the_week'] == 'Weekday'].select_dtypes(include=np.number).drop(columns=['order_id', 'customer_id'], errors='ignore')
            weekend_df = nyc_data[nyc_data['day_of_the_week'] == 'Weekend'].select_dtypes(include=np.number).drop(columns=['order_id', 'customer_id'], errors='ignore')

            col1, col2 = st.columns(2)
            with col1:
                weekday_corr = weekday_df.corr()
                fig_weekday = px.imshow(weekday_corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu', title="Weekday Correlation")
                st.plotly_chart(fig_weekday, use_container_width=True)
            with col2:
                weekend_corr = weekend_df.corr()
                fig_weekend = px.imshow(weekend_corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu', title="Weekend Correlation")
                st.plotly_chart(fig_weekend, use_container_width=True)
        
        with tab3:
            st.write("### Correlation by Cuisine Type")
            eastern_cuisines = ['Chinese', 'Korean', 'Japanese', 'Indian', 'Thai']
            western_cuisines = ['Italian', 'American', 'Mediterranean', 'Middle Eastern', 'Mexican', 'Southern', 'French', 'Spanish']
            
            # Drop IDs before correlation
            eastern_df = nyc_data[nyc_data['cuisine_type'].isin(eastern_cuisines)].select_dtypes(include=np.number).drop(columns=['order_id', 'customer_id'], errors='ignore')
            western_df = nyc_data[nyc_data['cuisine_type'].isin(western_cuisines)].select_dtypes(include=np.number).drop(columns=['order_id', 'customer_id'], errors='ignore')

            col1, col2 = st.columns(2)
            with col1:
                eastern_corr = eastern_df.corr()
                fig_eastern = px.imshow(eastern_corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu', title="Eastern Cuisine Correlation")
                st.plotly_chart(fig_eastern, use_container_width=True)
            with col2:
                western_corr = western_df.corr()
                fig_western = px.imshow(western_corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu', title="Western Cuisine Correlation")
                st.plotly_chart(fig_western, use_container_width=True)
        
        with tab4:
            st.write("### Correlation Before and After Imputation")
            
            col1, col2 = st.columns(2)

            with col1:
                st.write("#### Before KNN Imputation")
                # Use the original numeric data with NaNs
                pre_imputation_corr = nyc_numeric.corr()
                fig_before = px.imshow(pre_imputation_corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu', title="Correlation (Before Imputation)")
                st.plotly_chart(fig_before, use_container_width=True)

            with col2:
                st.write("#### After KNN Imputation")
                # Perform KNN Imputation
                imputer = KNNImputer(n_neighbors=5)
                nyc_imputed_array = imputer.fit_transform(nyc_numeric)
                # Create a new dataframe with the imputed values
                nyc_imputed_df = pd.DataFrame(nyc_imputed_array, columns=nyc_numeric.columns)
                
                post_imputation_corr = nyc_imputed_df.corr()
                fig_after = px.imshow(post_imputation_corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu', title="Correlation (After Imputation)")
                st.plotly_chart(fig_after, use_container_width=True)

else:
    st.info("Please select a dataset from the sidebar to begin analysis.")

st.sidebar.info("Dashboard created based on the project files.")
