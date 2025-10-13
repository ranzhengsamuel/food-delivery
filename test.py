import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.impute import KNNImputer

# Set page config
st.set_page_config(page_title="Delivery Time Analysis", layout="wide")

# Title and introduction
st.title("Food Delivery Time Factor Analysis")
st.markdown(
    "This app explores factors affecting food delivery times using two datasets. Use the controls below to interact with the analysis.")

# ----------------------
# Data Loading Section
# ----------------------
st.sidebar.header("Data Loading")
st.sidebar.markdown("Ensure CSV files are in the same directory as this app")


@st.cache_data
def load_data():
    try:
        nyc = pd.read_csv("NYC_food_order.csv", na_values=['Not given'])
        fdt = pd.read_csv("Food_Delivery_Times.csv")
        # Drop first two columns from NYC data as in original analysis
        nyc = nyc.drop(nyc.columns[:2], axis=1)
        return nyc, fdt
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Please check file paths.")
        return None, None


nyc, fdt = load_data()

if nyc is not None and fdt is not None:
    # Show basic data info
    with st.expander("View Dataset Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("NYC Food Order Data")
            st.write(f"Shape: {nyc.shape}")
            st.dataframe(nyc.head())
        with col2:
            st.subheader("Food Delivery Times Data")
            st.write(f"Shape: {fdt.shape}")
            st.dataframe(fdt.head())

    # ----------------------
    # Missing Values Analysis
    # ----------------------
    st.header("Missing Values Analysis")
    missing_analysis = st.sidebar.checkbox("Show missing values heatmaps", True)

    if missing_analysis:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("NYC Data Missing Values")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(nyc.isna(), cmap="magma", ax=ax)
            ax.set_title("Missing Values Heatmap (NYC)")
            st.pyplot(fig)

        with col2:
            st.subheader("Delivery Times Missing Values")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(fdt.isna(), cmap="magma", ax=ax)
            ax.set_title("Missing Values Heatmap (Delivery Times)")
            st.pyplot(fig)

    # ----------------------
    # Data Imputation
    # ----------------------
    st.header("Data Imputation with KNN")
    impute_data = st.sidebar.checkbox("Apply KNN imputation", True)

    if impute_data:
        # Process NYC data
        nyc_numeric_cols = nyc.select_dtypes(include=['number']).columns.tolist()
        imputer = KNNImputer(n_neighbors=5)
        nyc[nyc_numeric_cols] = imputer.fit_transform(nyc[nyc_numeric_cols])

        # Process FDT data (fixed the original nyc reference error)
        fdt_numeric_cols = fdt.select_dtypes(include=['number']).columns.tolist()
        fdt[fdt_numeric_cols] = imputer.fit_transform(fdt[fdt_numeric_cols])

        st.success("KNN imputation applied to numeric columns in both datasets")
        with st.expander("View imputation results"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("NYC data after imputation (missing values count):")
                st.dataframe(nyc[nyc_numeric_cols].isna().sum().to_frame().T)
            with col2:
                st.write("Delivery Times data after imputation (missing values count):")
                st.dataframe(fdt[fdt_numeric_cols].isna().sum().to_frame().T)

    # ----------------------
    # NYC Dataset Analysis
    # ----------------------
    st.header("NYC Food Order Analysis")
    nyc_analysis = st.sidebar.checkbox("Show NYC dataset analysis", True)

    if nyc_analysis:
        # Correlation analysis controls
        time_filter = st.sidebar.radio(
            "Filter by day type (NYC data):",
            ["All", "Weekday", "Weekend"]
        )

        cuisine_filter = st.sidebar.radio(
            "Filter by cuisine type (NYC data):",
            ["All", "Eastern", "Western"]
        )

        # Prepare data based on filters
        nyc_numeric = nyc.select_dtypes(include='number')
        eastern_cuisines = ['Chinese', 'Korean', 'Japanese', 'Indian', 'Thai']
        western_cuisines = ['Italian', 'American', 'Mediterranean', 'Middle Eastern',
                            'Mexican', 'Southern', 'French', 'Spanish']

        if time_filter == "Weekday":
            filtered_nyc = nyc[nyc['day_of_the_week'] == 'Weekday']
        elif time_filter == "Weekend":
            filtered_nyc = nyc[nyc['day_of_the_week'] == 'Weekend']
        else:
            filtered_nyc = nyc

        if cuisine_filter == "Eastern":
            filtered_nyc = filtered_nyc[filtered_nyc['cuisine_type'].isin(eastern_cuisines)]
        elif cuisine_filter == "Western":
            filtered_nyc = filtered_nyc[filtered_nyc['cuisine_type'].isin(western_cuisines)]

        filtered_nyc_numeric = filtered_nyc.select_dtypes(include='number')

        # Correlation heatmap
        st.subheader(f"Correlation Heatmap ({time_filter} - {cuisine_filter} Cuisines)")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            filtered_nyc_numeric.corr(),
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            linewidths=0.5,
            ax=ax
        )
        ax.set_title(f'Correlation Heatmap ({time_filter} - {cuisine_filter})')
        st.pyplot(fig)

    # ----------------------
    # Delivery Times Analysis
    # ----------------------
    st.header("Food Delivery Times Analysis")
    fdt_analysis = st.sidebar.checkbox("Show delivery times analysis", True)

    if fdt_analysis:
        # Correlation heatmap
        st.subheader("Overall Correlation Heatmap (Delivery Times)")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            fdt.select_dtypes(include='number').corr(),
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            linewidths=0.5,
            ax=ax
        )
        st.pyplot(fig)

        # Categorical analysis
        st.subheader("Delivery Time vs Categorical Variables")
        cat_var = st.sidebar.selectbox(
            "Select categorical variable to analyze:",
            ['Time_of_Day', 'Vehicle_Type', 'Traffic_Level', 'Weather']
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=cat_var, y='Delivery_Time_min', data=fdt, ax=ax)
        ax.set_title(f'Delivery Time vs {cat_var}')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Distance vs Delivery Time
        st.subheader("Distance vs Delivery Time Relationship")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=fdt, x='Distance_km', y='Delivery_Time_min', alpha=0.6, ax=ax)


        # Add regression line
        def linear_func(x, a, b):
            return a * x + b


        popt, pcov = curve_fit(linear_func, fdt['Distance_km'], fdt['Delivery_Time_min'])
        a, b = popt
        x_range = np.linspace(fdt['Distance_km'].min(), fdt['Distance_km'].max(), 100)
        y_fit = linear_func(x_range, a, b)
        ax.plot(x_range, y_fit, 'r-', linewidth=2, label=f'Line: y = {a:.1f}x + {b:.1f}')

        ax.set_title('Delivery Time vs Distance')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Delivery Time (min)')
        ax.legend()
        st.pyplot(fig)

        # Correlation stats
        correlation, p_value = pearsonr(fdt['Distance_km'], fdt['Delivery_Time_min'])
        st.write(f"Pearson correlation between Distance and Delivery Time: {correlation:.2f} (p-value: {p_value:.4f})")

        # Distance bin analysis
        st.subheader("Delivery Time by Distance Bins")
        bin_size = st.sidebar.slider("Select distance bin size (km)", 1, 10, 5)
        fdt['distance_bin'] = pd.cut(
            fdt['Distance_km'],
            bins=range(0, int(fdt['Distance_km'].max()) + bin_size, bin_size)
        )
        bin_stats = fdt.groupby('distance_bin')['Delivery_Time_min'].agg(['mean', 'median', 'std']).reset_index()
        st.dataframe(bin_stats)

        # Plot bin stats
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='distance_bin', y='mean', data=bin_stats, ax=ax)
        ax.set_title(f'Mean Delivery Time by {bin_size}km Distance Bins')
        plt.xticks(rotation=45)
        ax.set_ylabel('Mean Delivery Time (min)')
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Analysis based on NYC food order and delivery time datasets")