
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(layout="wide", page_title="Food Delivery Time Prediction Dashboard")

# Title and description
st.title("Food Delivery Time Prediction Dashboard")
st.write("An interactive dashboard for Initial Data Analysis (IDA) and Exploratory Data Analysis (EDA) of the Food Delivery Time Prediction dataset.")

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads the dataset from a local CSV file."""
    try:
        df = pd.read_csv(r"C:\Users\USER\Desktop\MSU related\Courses Sem 1\CMSE 830\Project\data\Food_Delivery_Times.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()

if data is not None:
    # --- IDA Section ---
    st.header("Initial Data Analysis (IDA)")

    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(data.head())

    # Display data shape and column types
    st.subheader("Data Shape and Column Types")
    st.write("Data Shape:", data.shape)
    st.write("Column Types:", data.dtypes)

    # --- EDA Section ---
    st.header("Exploratory Data Analysis (EDA)")

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    selected_city = st.sidebar.multiselect("City", data["City"].unique(), default=data["City"].unique())
    selected_weather = st.sidebar.multiselect("Weather Conditions", data["Weatherconditions"].unique(), default=data["Weatherconditions"].unique())
    selected_traffic = st.sidebar.multiselect("Road Traffic Density", data["Road_traffic_density"].unique(), default=data["Road_traffic_density"].unique())

    # Filter data based on selections
    filtered_data = data[
        (data["City"].isin(selected_city)) &
        (data["Weatherconditions"].isin(selected_weather)) &
        (data["Road_traffic_density"].isin(selected_traffic))
    ]

    st.subheader("Filtered Data")
    st.dataframe(filtered_data.head())

    # --- Visualizations ---
    st.subheader("Data Visualizations")

    # Distribution of Delivery Time
    fig_delivery_time = px.histogram(filtered_data, x="Time_taken(min)", nbins=30, title="Distribution of Delivery Time (min)")
    st.plotly_chart(fig_delivery_time)

    # Delivery Time by Weather Conditions
    fig_weather = px.box(filtered_data, x="Weatherconditions", y="Time_taken(min)", title="Delivery Time by Weather Conditions")
    st.plotly_chart(fig_weather)

    # Delivery Time by Road Traffic Density
    fig_traffic = px.box(filtered_data, x="Road_traffic_density", y="Time_taken(min)", title="Delivery Time by Road Traffic Density")
    st.plotly_chart(fig_traffic)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    # Label encode categorical features for correlation calculation
    le = LabelEncoder()
    corr_data = data.copy()
    for col in corr_data.columns:
        if corr_data[col].dtype == 'object':
            corr_data[col] = le.fit_transform(corr_data[col])

    corr = corr_data.corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis'
    ))
    fig_corr.update_layout(title="Correlation Heatmap of Features")
    st.plotly_chart(fig_corr)

    # Scatter plot of Delivery Person's Age vs. Delivery Time
    st.subheader("Delivery Person's Age vs. Delivery Time")
    fig_age_time = px.scatter(filtered_data, x="Delivery_person_Age", y="Time_taken(min)",
                              title="Delivery Person's Age vs. Delivery Time",
                              trendline="ols")
    st.plotly_chart(fig_age_time)

    # Scatter plot of Delivery Person's Ratings vs. Delivery Time
    st.subheader("Delivery Person's Ratings vs. Delivery Time")
    fig_ratings_time = px.scatter(filtered_data, x="Delivery_person_Ratings", y="Time_taken(min)",
                                  title="Delivery Person's Ratings vs. Delivery Time",
                                  trendline="ols")
    st.plotly_chart(fig_ratings_time)

else:
    st.warning("Could not load data. Please check your internet connection and Kaggle API credentials.")
