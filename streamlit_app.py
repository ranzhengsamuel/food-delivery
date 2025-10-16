
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.impute import KNNImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(layout="wide", page_title="Food Delivery Analysis Dashboard")

# Title and description
st.title("What affects food delivery time?")
st.write("People nowadays order more and more takeouts. Of course, we all want the order to arrive ASAP. But what really affects the time of delivery? Is it weather? Is it distance? Is it the quality of the restaurant? Let's find out!")

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
    ("Food Delivery Times", "NYC Food Orders", "Comparison of the 2")
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

            st.subheader("Data Summary")
            with st.expander("Data Info"):
                buffer = io.StringIO()
                filtered_data.info(buf=buffer)
                st.text(buffer.getvalue())
            with st.expander("Descriptive Statistics"):
                st.dataframe(filtered_data.describe())

            # --- Visualizations ---
            st.subheader("Data Visualizations")

            # Missing Value Heatmap
            st.write("### Missing Value Heatmap")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(filtered_data.isnull().T, cbar=False, cmap='viridis', ax=ax)
            st.pyplot(fig)

            # 1. Distribution of Delivery Time
            st.write("### Distribution of Delivery Time")
            
            # Create a distribution plot with a KDE curve
            hist_data = [filtered_data["Delivery_Time_min"].dropna()]
            group_labels = ['Delivery Time'] # name of the dataset
            
            fig_delivery_time = ff.create_distplot(hist_data, group_labels, bin_size=1, show_rug=False)
            fig_delivery_time.update_layout(title_text='Distribution of Delivery Time with KDE')
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

            # Pie charts of distance based on road traffic density and weather
            st.subheader("Distance Category Analysis")
            bins = [0, 5, 10, 15, float('inf')]
            labels = ['0-5km', '5-10km', '10-15km', '15km+']
            data['distance_category'] = pd.cut(data['Distance_km'], bins=bins, labels=labels, right=False)

            col1, col2 = st.columns(2)

            with col1:
                st.write("#### By Road Traffic Density")
                traffic_density_option = st.selectbox(
                    'Select Road Traffic Density',
                    data['Traffic_Level'].dropna().unique(),
                    key='traffic_select'
                )
                filtered_by_traffic_pie = data[data['Traffic_Level'] == traffic_density_option]
                fig_pie_traffic = px.pie(
                    filtered_by_traffic_pie, 
                    names='distance_category', 
                    title=f'Distance Categories for {traffic_density_option} Traffic'
                )
                st.plotly_chart(fig_pie_traffic, use_container_width=True)

            with col2:
                st.write("#### By Weather Condition")
                weather_condition_option = st.selectbox(
                    'Select Weather Condition',
                    data['Weather'].dropna().unique(),
                    key='weather_select'
                )
                filtered_by_weather_pie = data[data['Weather'] == weather_condition_option]
                fig_pie_weather = px.pie(
                    filtered_by_weather_pie, 
                    names='distance_category', 
                    title=f'Distance Categories for {weather_condition_option} Weather'
                )
                st.plotly_chart(fig_pie_weather, use_container_width=True)

            # 4. Scatter Plots for Numerical Analysis
            st.write("### Numerical Feature Analysis")
            
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

            st.markdown("""
            ### Conclusion
            From this dataset, we came to the conclusion that the only factor that affects the food delivery time is distance, which can roughly be calculated by the line y = 3x+26.3. This is based on cities that are not major metropolis. We will see if other factors play a role in determining delivery time in major metropolis like NYC in the next page.
            """)

        else:
            st.warning("The 'Food Delivery Times' dataset is missing one or more required columns: 'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type'.")


elif dataset_choice == "NYC Food Orders":
    st.header("Analysis of NYC Food Orders")
    nyc_data = load_data("NYC_food_order.csv")

    if nyc_data is not None:
        # --- Data Overview ---
        st.subheader("Data Preview")
        st.dataframe(nyc_data.head())

        st.subheader("Data Summary")
        with st.expander("Data Info"):
            buffer = io.StringIO()
            nyc_data.info(buf=buffer)
            st.text(buffer.getvalue())
        with st.expander("Descriptive Statistics"):
            st.dataframe(nyc_data.describe())

        # --- Visualizations ---
        st.subheader("Data Visualizations")
        
        # Missing Value Heatmap
        st.write("### Missing Value Heatmap")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(nyc_data.isnull().T, cbar=False, cmap='viridis', ax=ax)
        st.pyplot(fig)

        fig_delivery_dist = px.histogram(nyc_data, x="delivery_time", nbins=20, title="Distribution of Delivery Time (min)")
        st.plotly_chart(fig_delivery_dist, use_container_width=True)

        # Pre-process data, dropping ID columns for correlation
        nyc_numeric = nyc_data.select_dtypes(include=np.number).drop(columns=['order_id', 'customer_id'], errors='ignore')
        
        st.subheader("Imputation Analysis")
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

        st.subheader("Correlation Heatmap Analysis")
        st.write("Exploring correlations within the NYC Food Order dataset based on different segmentations.")

        # --- Tabs for different views ---
        tab1, tab2, tab3 = st.tabs(["Overall Correlation", "Weekday vs. Weekend", "Cuisine Type"])

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

elif dataset_choice == "Comparison of the 2":
    st.header("Comparison of Delivery Times")

    # Load both datasets
    food_delivery_data = load_data("Food_Delivery_Times.csv")
    nyc_data = load_data("NYC_food_order.csv")

    if food_delivery_data is not None and nyc_data is not None:
        # Create a figure for the box plots
        fig = go.Figure()

        # Add box plot for Food Delivery Times
        fig.add_trace(go.Box(
            y=food_delivery_data["Delivery_Time_min"],
            name="Food Delivery Times",
            marker_color='indianred'
        ))

        # Add box plot for NYC Food Orders
        fig.add_trace(go.Box(
            y=nyc_data["delivery_time"],
            name="NYC Food Orders",
            marker_color='lightseagreen'
        ))

        fig.update_layout(
            title="Comparison of Delivery Times",
            yaxis_title="Delivery Time (minutes)",
            boxmode='group'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        From these boxplots, we can see that the delivery times in the Food Delivery Times dataset is significantly longer than in NYC. This is because the data collected in that dataset is mostly in non-major towns and cities, and therefore having significantly longer distance, and thus resulting in much longer delivery time
        """)
    else:
        st.error("Could not load one or both datasets for comparison.")

else:
    st.info("Please select a dataset from the sidebar to begin analysis.")

st.sidebar.info("Dashboard created based on the project files.")
