import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from sklearn.impute import KNNImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

st.set_page_config(layout="wide", page_title="Food Delivery Analysis Dashboard")

st.title("What affects food delivery time?")
st.write("People nowadays order more and more takeouts. Of course, we all want the order to arrive ASAP. But what really affects the time of delivery? Is it weather? Is it distance? Is it the quality of the restaurant? Let's find out!")

@st.cache_data
def load_data(file_name):
    try:
        data_path = os.path.join("data", file_name)
        if not os.path.exists(data_path):
            data_path = file_name
        df = pd.read_csv(data_path, na_values=['Not given', 'NaN ', 'nan', 'NAN', 'Nan', None, '', ' '])
        return df
    except Exception as e:
        st.error(f"Error loading data from {file_name}: {e}")
        return None

st.sidebar.header("Dataset Selection")
dataset_choice = st.sidebar.radio(
    "Choose a dataset to analyze:",
    ("About the Dataset", "Food Delivery Times", "NYC Food Orders", "Indian Food Delivery", "Overall Conclusion and Findings")
)

if dataset_choice == "About the Dataset":
    st.header("📊 About the Dataset")

    st.markdown("""
    ## Overview
    There are 3 datasets being used, among them, **"Food Delivery Times", "NYC Food Orders" are made from Mid-term Project, whereas "Indian Food Delivery" are mostly End Semester project content.**

    ### Datasets Used
    """)

    # Dataset 1
    st.subheader("1. Food Delivery Times (mid-term)")
    st.markdown("""
    - **Source:** [Kaggle]
    - **Size:** [Around 1000]
    - **Key Features:**
        - Weather conditions
        - Traffic level
        - Time of day
        - Vehicle type
        - Distance (km)
        - Delivery time (minutes)
    - **Focus:** Understanding the relationship between distance and delivery time in non-metropolitan areas
    """)

    # Dataset 2
    st.subheader("2. NYC Food Orders (mid-term)")
    st.markdown("""
    - **Source:** [Kaggle]
    - **Size:** [About 1900]
    - **Key Features:**
        - Order details
        - Customer information
        - Cuisine type
        - Day of the week
        - Delivery time
    - **Focus:** Analyzing delivery patterns in a major metropolitan area (New York City)
    - **Special Notes:** Contains missing values that were handled using KNN imputation
    """)

    # Dataset 3
    st.subheader("3. Indian Food Delivery (End of Semester)")
    st.markdown("""
    - **Source:** [Kaggle]
    - **Size:** [About 44500]
    - **Key Features:**
        - Restaurant and delivery location coordinates
        - Vehicle condition
        - Road traffic density
        - Weather conditions
        - Delivery time (minutes)
    - **Focus:** Comprehensive analysis of traffic, weather, and vehicle factors
    - **Special Notes:** Did thorough EDA, Data Processing/Feature Engineering, as well as Machine Learning for evaluating the model
    """)

if dataset_choice == "Food Delivery Times":
    st.header("Analysis of Food Delivery Times")
    data = load_data("Food_Delivery_Times.csv")

    if data is not None:
        st.sidebar.header("Filters")
        data.columns = [col.strip() for col in data.columns]
        required_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
        if all(col in data.columns for col in required_cols):
            weather_options = data["Weather"].dropna().unique()
            traffic_options = data["Traffic_Level"].dropna().unique()
            selected_weather = st.sidebar.multiselect("Weather Conditions", weather_options, default=weather_options)
            selected_traffic = st.sidebar.multiselect("Road Traffic Density", traffic_options, default=traffic_options)
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

            st.subheader("Data Visualizations")

            st.write("### Missing Value Heatmap")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(filtered_data.isnull().T, cbar=False, cmap='viridis', ax=ax)
            st.pyplot(fig)

            st.write("### Missing Value Analysis")

            total_cells = filtered_data.shape[0] * filtered_data.shape[1]
            missing_cells = filtered_data.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Missing Values", f"{missing_cells:,}")

            with col2:
                st.metric("Percentage of Total", f"{missing_percentage:.2f}%")

            with col3:
                st.metric("Rows with Missing Values", filtered_data.isnull().any(axis=1).sum())

            st.write("#### Missing Values by Column")
            missing_by_column = filtered_data.isnull().sum()
            missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)

            if len(missing_by_column) > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_by_column.index,
                    'Missing Count': missing_by_column.values,
                    'Percentage': (missing_by_column.values / filtered_data.shape[0] * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True)

                # Drop missing values
                st.write("#### Action: Dropping Missing Values")
                filtered_data_clean = filtered_data.dropna()

                rows_before = filtered_data.shape[0]
                rows_after = filtered_data_clean.shape[0]
                rows_dropped = rows_before - rows_after

                st.info(f"""
                **Data Cleaning Summary:**
                - Rows before: {rows_before:,}
                - Rows after: {rows_after:,}
                - Rows dropped: {rows_dropped:,} ({(rows_dropped / rows_before * 100):.2f}%)
                - Type of Missingness after Analysis: MCAR
                - Reason for dropping: 1. the percentage is really small, only 6%. 2. Since it is MCAR, it does not affect the original correlation
                """)

                filtered_data = filtered_data_clean
            else:
                st.success("✅ No missing values found in the filtered dataset!")

            st.write("### Distribution of Delivery Time")
            
            hist_data = [filtered_data["Delivery_Time_min"].dropna()]
            group_labels = ['Delivery Time'] 
            
            fig_delivery_time = ff.create_distplot(hist_data, group_labels, bin_size=1, show_rug=False)
            fig_delivery_time.update_layout(title_text='Distribution of Delivery Time with KDE')
            st.plotly_chart(fig_delivery_time, use_container_width=True)

            st.write("### Correlation Heatmap of Numerical Features")
            numeric_cols = filtered_data.select_dtypes(include=['number']).drop(columns=['Order_ID'], errors='ignore')
            corr = numeric_cols.corr()
            fig_corr = px.imshow(corr, text_auto='.2f', aspect="auto", color_continuous_scale='Viridis', title="Correlation Heatmap of Numerical Features")
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.write("### Categorical Feature Analysis")
            col1, col2 = st.columns(2)

            with col1:
                fig_weather = px.box(filtered_data, x="Weather", y="Delivery_Time_min", title="Delivery Time by Weather Conditions")
                st.plotly_chart(fig_weather, use_container_width=True)
                
                fig_time_of_day = px.box(filtered_data, x="Time_of_Day", y="Delivery_Time_min", title="Delivery Time vs. Time of Day")
                st.plotly_chart(fig_time_of_day, use_container_width=True)

            with col2:
                fig_traffic = px.box(filtered_data, x="Traffic_Level", y="Delivery_Time_min", title="Delivery Time by Road Traffic Density")
                st.plotly_chart(fig_traffic, use_container_width=True)

                fig_vehicle_type = px.box(filtered_data, x="Vehicle_Type", y="Delivery_Time_min", title="Delivery Time vs. Vehicle Type")
                st.plotly_chart(fig_vehicle_type, use_container_width=True)

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

            st.write("### Numerical Feature Analysis")
            
            fig_ratings_time = px.scatter(filtered_data, x="Distance_km", y="Delivery_Time_min",
                                          title="Distance (km) vs. Delivery Time",
                                          trendline="ols")
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
        st.subheader("Data Preview")
        st.dataframe(nyc_data.head())

        st.subheader("Data Summary")
        with st.expander("Data Info"):
            buffer = io.StringIO()
            nyc_data.info(buf=buffer)
            st.text(buffer.getvalue())
        with st.expander("Descriptive Statistics"):
            st.dataframe(nyc_data.describe())

        st.subheader("Data Visualizations")
        
        st.write("### Missing Value Heatmap")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(nyc_data.isnull().T, cbar=False, cmap='viridis', ax=ax)
        st.pyplot(fig)

        fig_delivery_dist = px.histogram(nyc_data, x="delivery_time", nbins=20, title="Distribution of Delivery Time (min)")
        st.plotly_chart(fig_delivery_dist, use_container_width=True)

        nyc_numeric = nyc_data.select_dtypes(include=np.number).drop(columns=['order_id', 'customer_id'], errors='ignore')
        
        st.subheader("Imputation Analysis")
        st.write("### Correlation Before and After Imputation")
        
        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Before KNN Imputation")
            pre_imputation_corr = nyc_numeric.corr()
            fig_before = px.imshow(pre_imputation_corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu', title="Correlation (Before Imputation)")
            st.plotly_chart(fig_before, use_container_width=True)

        with col2:
            st.write("#### After KNN Imputation")
            imputer = KNNImputer(n_neighbors=5)
            nyc_imputed_array = imputer.fit_transform(nyc_numeric)
            nyc_imputed_df = pd.DataFrame(nyc_imputed_array, columns=nyc_numeric.columns)
            
            post_imputation_corr = nyc_imputed_df.corr()
            fig_after = px.imshow(post_imputation_corr, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu', title="Correlation (After Imputation)")
            st.plotly_chart(fig_after, use_container_width=True)

        st.subheader("Correlation Heatmap Analysis")
        st.write("Exploring correlations within the NYC Food Order dataset based on different segmentations.")

        tab1, tab2, tab3 = st.tabs(["Overall Correlation", "Weekday vs. Weekend", "Cuisine Type"])

        with tab1:
            st.write("### Overall Numeric Correlation")
            corr_matrix = nyc_numeric.corr()
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu', title="Overall Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.write("### Correlation by Day of the Week")
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

elif dataset_choice == "Indian Food Delivery":
    st.header("Analysis of Indian Food Delivery")

    ifd_page = st.sidebar.radio(
        "Select Analysis Section:",
        [
            "Initial Data Analysis",
            "Missing Value Analysis",
            "Linear Regression: Distance vs Time",
            "Correlation Analysis",
            "Vehicle Condition & Distance Analysis",
            "Distribution by Vehicle/Weather/Traffic",
            "Feature Engineering & Models",
        ]
    )

    ifd_data = load_data("India_Food_Delivery.csv")

    if ifd_data is not None:

        # ==================== Initial Data Analysis ====================
        if ifd_page == "Initial Data Analysis":
            st.subheader("Initial Data Analysis")

            with st.spinner('Preparing data...'):
                ifd_fixed = ifd_data.dropna().copy()

                ifd_fixed['Restaurant_latitude'] = ifd_fixed['Restaurant_latitude'].abs()
                ifd_fixed['Restaurant_longitude'] = ifd_fixed['Restaurant_longitude'].abs()
                ifd_fixed['Delivery_location_latitude'] = ifd_fixed['Delivery_location_latitude'].abs()
                ifd_fixed['Delivery_location_longitude'] = ifd_fixed['Delivery_location_longitude'].abs()


                # calculate distance
                @st.cache_data
                def calculate_distance(df):
                    def dist_calc(row):
                        restaurant = (row['Restaurant_latitude'], row['Restaurant_longitude'])
                        delivery = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
                        return geodesic(restaurant, delivery).kilometers

                    df['distance_km'] = df.apply(dist_calc, axis=1)
                    return df


                ifd_fixed = calculate_distance(ifd_fixed)

            st.write("### Cleaned Data Preview")
            st.dataframe(ifd_fixed.head())

            st.write("### Data Summary")
            with st.expander("Data Info (Cleaned)"):
                buffer = io.StringIO()
                ifd_fixed.info(buf=buffer)
                st.text(buffer.getvalue())

            with st.expander("Descriptive Statistics (Cleaned)"):
                st.dataframe(ifd_fixed.describe())

        # ==================== Missing Value Analysis ====================
        elif ifd_page == "Missing Value Analysis":
            st.subheader("Missing Value Analysis")
            st.write(
                "We will look at how many missing values there are first. Since there is more than enough data we can work on, we will drop all of the missing value rows.")

            st.write("### Missing Values Before Cleaning")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(ifd_data.isnull().T, cbar=False, cmap='viridis', ax=ax)
            st.pyplot(fig)

            # missingness
            total_cells = ifd_data.shape[0] * ifd_data.shape[1]
            missing_cells = ifd_data.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Missing Values", f"{missing_cells:,}")

            with col2:
                st.metric("Percentage of Total", f"{missing_percentage:.2f}%")

            with col3:
                st.metric("Rows with Missing Values", ifd_data.isnull().any(axis=1).sum())

            st.write("#### Missing Values by Column")
            missing_by_column = ifd_data.isnull().sum()
            missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)

            if len(missing_by_column) > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_by_column.index,
                    'Missing Count': missing_by_column.values,
                    'Percentage': (missing_by_column.values / ifd_data.shape[0] * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True)

            # Drop missing values
            st.write("#### Action: Dropping Missing Values")

            rows_before = ifd_data.shape[0]
            ifd_cleaned = ifd_data.dropna()
            rows_after = ifd_cleaned.shape[0]
            rows_dropped = rows_before - rows_after

            st.info(f"""
            **Data Cleaning Summary:**
            - Rows before: {rows_before:,}
            - Rows after: {rows_after:,}
            - Rows dropped: {rows_dropped:,} ({(rows_dropped / rows_before * 100):.2f}%)
            - Retained data: {(rows_after / rows_before * 100):.2f}%
            - Type of Missingness: MAR (Delivery_person_Ratings missing correlates with order time)
            - Reason for dropping: 1. Although MAR, the columns that are missing do not affect the main columns for correlation analysis. 2. There is enough data for analysis (90.73% retained)
            """)

            st.write("### Missing Values After Cleaning")
            fig_after, ax_after = plt.subplots(figsize=(10, 4))
            sns.heatmap(ifd_cleaned.isnull().T, cbar=False, cmap='viridis', ax=ax_after)
            st.pyplot(fig_after)

        # ==================== Linear Regression ====================
        elif ifd_page == "Linear Regression: Distance vs Time":
            st.subheader("Linear Regression: Distance vs. Delivery Time")

            with st.spinner('Calculating distances...'):
                # prepare data
                ifd_fixed = ifd_data.dropna().copy()


                # calculate original distance
                def calculate_distance_original(df):
                    def dist_calc(row):
                        restaurant = (row['Restaurant_latitude'], row['Restaurant_longitude'])
                        delivery = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
                        return geodesic(restaurant, delivery).kilometers

                    df['distance_km'] = df.apply(dist_calc, axis=1)
                    return df


                ifd_fixed = calculate_distance_original(ifd_fixed)

            # diagnosis
            with st.expander("🔍 Data Quality Diagnosis"):
                st.write("**Distance Statistics (Before Correction):**")
                st.write(ifd_fixed['distance_km'].describe())

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Distance = 0", (ifd_fixed['distance_km'] == 0).sum())
                with col2:
                    st.metric("Distance > 100km", (ifd_fixed['distance_km'] > 100).sum())
                with col3:
                    st.metric("Distance > 1000km", (ifd_fixed['distance_km'] > 1000).sum())

                st.warning(
                    "⚠️ Found records with distance > 100km. This indicates negative latitude/longitude values in the original dataset.")

            st.write(
                "Upon initial analysis, the linear regression plot appeared absurd. This was due to the presence of negative coordinate values in the dataset.")

            st.write("#### Original Plot (Before Correction)")

            # fix 1st graph
            slope_original, intercept_original, r_value_original, p_value_original, std_err_original = stats.linregress(
                ifd_fixed['distance_km'],
                ifd_fixed['Time_taken(min)']
            )
            plot_data_original = ifd_fixed[['distance_km', 'Time_taken(min)']].copy()
            fig_linreg_before = px.scatter(
                plot_data_original,
                x='distance_km',
                y='Time_taken(min)',
                title=f"Original Data with Negative Coordinates (R² = {r_value_original ** 2:.4f})",
                trendline="ols",
                labels={'distance_km': 'Distance (km)', 'Time_taken(min)': 'Time taken (min)'}
            )
            st.plotly_chart(fig_linreg_before, use_container_width=True)

            st.write(
                "**This is because some latitude value is negative, resulting in significantly large distance values!!!** To correct this, the absolute values of the **latitude and longitude coordinates** were applied, then the distance was recalculated.")

            st.write("#### Corrected Plot (After Applying Absolute Values)")

            with st.spinner('Recalculating with corrected coordinates...'):
                # fix the coordinate
                ifd_corrected = ifd_fixed.copy()
                ifd_corrected['Restaurant_latitude'] = ifd_corrected['Restaurant_latitude'].abs()
                ifd_corrected['Restaurant_longitude'] = ifd_corrected['Restaurant_longitude'].abs()
                ifd_corrected['Delivery_location_latitude'] = ifd_corrected['Delivery_location_latitude'].abs()
                ifd_corrected['Delivery_location_longitude'] = ifd_corrected['Delivery_location_longitude'].abs()


                def calculate_distance_corrected(df):
                    def dist_calc(row):
                        restaurant = (row['Restaurant_latitude'], row['Restaurant_longitude'])
                        delivery = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
                        return geodesic(restaurant, delivery).kilometers

                    df['distance_km'] = df.apply(dist_calc, axis=1)
                    return df


                ifd_corrected = calculate_distance_corrected(ifd_corrected)

            # second picture
            slope_corrected, intercept_corrected, r_value_corrected, p_value_corrected, std_err_corrected = stats.linregress(
                ifd_corrected['distance_km'],
                ifd_corrected['Time_taken(min)']
            )

            plot_data_corrected = ifd_corrected[['distance_km', 'Time_taken(min)']].copy()
            plot_data_corrected['distance_km'] = plot_data_corrected['distance_km'].astype(float)

            fig_linreg_after = px.scatter(
                plot_data_corrected,  # only continuous
                x='distance_km',
                y='Time_taken(min)',
                title=f"Corrected Data with Fixed Coordinates (R² = {r_value_corrected ** 2:.4f})",
                trendline="ols",
                labels={'distance_km': 'Distance (km)', 'Time_taken(min)': 'Time taken (min)'}
            )
            fig_linreg_after.update_xaxes(type='linear')

            fig_linreg_after.add_annotation(
                x=0.05, y=0.95,
                xref="paper", yref="paper",
                text=f"y = {slope_corrected:.2f}x + {intercept_corrected:.2f}",
                showarrow=False,
                font=dict(size=14, color="black"),
                bgcolor="rgba(255, 255, 255, 0.5)"
            )

            st.plotly_chart(fig_linreg_after, use_container_width=True)

            # compare
            st.write("#### Comparison: Before vs After Correction")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original R²", f"{r_value_original ** 2:.4f}")
                st.metric("Original Distance > 100km", f"{(ifd_fixed['distance_km'] > 100).sum()}")
            with col2:
                st.metric("Corrected R²", f"{r_value_corrected ** 2:.4f}",
                          delta=f"+{(r_value_corrected ** 2 - r_value_original ** 2):.4f}")
                st.metric("Corrected Distance > 100km", f"{(ifd_corrected['distance_km'] > 100).sum()}")

            st.info(f"""
            **Overall Linear Regression Statistics:**
            - Equation: y = {slope_corrected:.3f}x + {intercept_corrected:.2f}
            - R² = {r_value_corrected ** 2:.3f}
            - p-value < 0.001

            💡 **Insight:** R² = {r_value_corrected ** 2:.3f} means {r_value_corrected ** 2 * 100:.1f}% of the variance in delivery time can be explained by distance. 
            The remaining variance is likely due to other factors like vehicle condition, traffic density, and weather.
            """)

        # ==================== Correlation Analysis ====================
        elif ifd_page == "Correlation Analysis":
            st.subheader("Correlation Analysis")

            with st.spinner('Preparing data...'):
                ifd_corrected = ifd_data.dropna().copy()
                ifd_corrected['Restaurant_latitude'] = ifd_corrected['Restaurant_latitude'].abs()
                ifd_corrected['Restaurant_longitude'] = ifd_corrected['Restaurant_longitude'].abs()
                ifd_corrected['Delivery_location_latitude'] = ifd_corrected['Delivery_location_latitude'].abs()
                ifd_corrected['Delivery_location_longitude'] = ifd_corrected['Delivery_location_longitude'].abs()


                @st.cache_data
                def calculate_distance(df):
                    def dist_calc(row):
                        restaurant = (row['Restaurant_latitude'], row['Restaurant_longitude'])
                        delivery = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
                        return geodesic(restaurant, delivery).kilometers

                    df['distance_km'] = df.apply(dist_calc, axis=1)
                    return df


                ifd_corrected = calculate_distance(ifd_corrected)

            st.write(
                "We are particularly interested in Vehicle condition and distance column. For Road Traffic and Weather, since they are categorical, we will analyze them separately.")

            selected_cols = [
                'Restaurant_latitude',
                'Restaurant_longitude',
                'Delivery_location_latitude',
                'Delivery_location_longitude',
                'Vehicle_condition',
                'Time_taken(min)',
                'distance_km'
            ]

            available_cols = [col for col in selected_cols if col in ifd_corrected.columns]
            ifd_correlation_matrix = ifd_corrected[available_cols].corr()

            fig_corr_ifd = px.imshow(
                ifd_correlation_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0,
                title="Correlation Heatmap of Key Features (Corrected Data)",
                labels=dict(color="Correlation")
            )

            fig_corr_ifd.update_layout(width=800, height=700)
            st.plotly_chart(fig_corr_ifd, use_container_width=True)

            st.markdown("""
            ### Key Findings from Correlation Analysis
            From this graph, we can see that both distance and vehicle condition affects time taken, not a strong correlation but still significant enough for us to do some analysis. Next, we will look into them in depth.
            """)

        # ==================== Vehicle Condition & Distance Analysis ====================
        elif ifd_page == "Vehicle Condition & Distance Analysis":
            st.subheader("Delivery Time by Vehicle Condition and Distance")

            with st.spinner('Preparing data...'):
                ifd_corrected = ifd_data.dropna().copy()
                ifd_corrected['Restaurant_latitude'] = ifd_corrected['Restaurant_latitude'].abs()
                ifd_corrected['Restaurant_longitude'] = ifd_corrected['Restaurant_longitude'].abs()
                ifd_corrected['Delivery_location_latitude'] = ifd_corrected['Delivery_location_latitude'].abs()
                ifd_corrected['Delivery_location_longitude'] = ifd_corrected['Delivery_location_longitude'].abs()


                @st.cache_data
                def calculate_distance(df):
                    def dist_calc(row):
                        restaurant = (row['Restaurant_latitude'], row['Restaurant_longitude'])
                        delivery = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
                        return geodesic(restaurant, delivery).kilometers

                    df['distance_km'] = df.apply(dist_calc, axis=1)
                    return df


                ifd_corrected = calculate_distance(ifd_corrected)
                ifd_corrected['distance_bin'] = pd.cut(
                    ifd_corrected['distance_km'],
                    bins=5,
                    labels=['very short', 'short', 'medium', 'long', 'very long'],
                    ordered=True
                )

            st.write("### Box Plot: Distance Bins vs Vehicle Condition")
            fig_box_interaction = px.box(
                ifd_corrected,
                x='distance_bin',
                y='Time_taken(min)',
                color='Vehicle_condition',
                title='Delivery Time by Vehicle Condition and Distance Bins'
            )
            st.plotly_chart(fig_box_interaction, use_container_width=True)

            st.write("### Scatter Plot: Distance vs Time by Vehicle Condition")
            fig_scatter_strat = px.scatter(
                ifd_corrected,
                x='distance_km',
                y='Time_taken(min)',
                color='Vehicle_condition',
                title='Distance vs Time by Vehicle Condition (Stratified Analysis)',
                trendline="ols",
                labels={'distance_km': 'Distance (km)', 'Time_taken(min)': 'Time taken (min)'}
            )
            st.plotly_chart(fig_scatter_strat, use_container_width=True)

        # ==================== Distribution Analysis ====================
        elif ifd_page == "Distribution by Vehicle/Weather/Traffic":
            st.subheader("Delivery Time Distribution Analysis")

            with st.spinner('Preparing data...'):
                ifd_corrected = ifd_data.dropna().copy()
                ifd_corrected['Restaurant_latitude'] = ifd_corrected['Restaurant_latitude'].abs()
                ifd_corrected['Restaurant_longitude'] = ifd_corrected['Restaurant_longitude'].abs()
                ifd_corrected['Delivery_location_latitude'] = ifd_corrected['Delivery_location_latitude'].abs()
                ifd_corrected['Delivery_location_longitude'] = ifd_corrected['Delivery_location_longitude'].abs()


                @st.cache_data
                def calculate_distance(df):
                    def dist_calc(row):
                        restaurant = (row['Restaurant_latitude'], row['Restaurant_longitude'])
                        delivery = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
                        return geodesic(restaurant, delivery).kilometers

                    df['distance_km'] = df.apply(dist_calc, axis=1)
                    return df


                ifd_corrected = calculate_distance(ifd_corrected)

            st.write("### Delivery Time Distribution by Vehicle Type")
            fig_vehicle_type = px.box(
                ifd_corrected,
                x='Type_of_vehicle',
                y='Time_taken(min)',
                title='Delivery Time Distribution by Vehicle Type'
            )
            st.plotly_chart(fig_vehicle_type, use_container_width=True)

            st.markdown("""
            F stats: 524.8602  
            p value: 1.3894e-225 and Eta-squared (η²): 0.0268  
            Explain: Type_of_vehicle can explain Time_taken(min) standard deviation's 2.68%
            """)

            st.write("### Delivery Time Distribution by Road Traffic Density")
            fig_traffic_density = px.violin(
                ifd_corrected,
                x='Road_traffic_density',
                y='Time_taken(min)',
                title='Delivery Time Distribution by Road traffic density'
            )
            st.plotly_chart(fig_traffic_density, use_container_width=True)

            st.markdown("""
            F stats: 2895.9441  
            p value: 0.0000e+00 and Eta-squared (η²): 0.1858  
            Explain: Road_traffic_density can explain Time_taken(min) standard deviation's 18.58%
            """)

            st.write("### Average Delivery Time by Weather (Mean ± SD)")
            weather_summary = ifd_corrected.groupby('Weatherconditions')['Time_taken(min)'].agg(
                ['mean', 'std']).reset_index()

            fig_weather = px.bar(
                weather_summary,
                x='Weatherconditions',
                y='mean',
                error_y='std',
                title='Average Delivery Time by Weather (Mean ± SD)'
            )
            st.plotly_chart(fig_weather, use_container_width=True)

            st.markdown("""
            F stats: 608.2196  
            p value: 0.0000e+00 and Eta-squared (η²): 0.0642  
            Explain: Weather can explain Time_taken(min) standard deviation's 6.42%
            """)

        # ==================== Feature Engineering & Models ====================
        elif ifd_page == "Feature Engineering & Models":
            st.subheader("Data Processing and Feature Engineering")

            # add the second layer of radio
            model_section = st.radio(
                "Select Model Comparison:",
                [
                    "Overview",
                    "Model 1: Distance & Vehicle Condition",
                    "Model 2: Traffic & Weather Interactions",
                    "Conclusion: What Affects Delivery Time Most?"
                ]
            )

            with st.spinner('Preparing data...'):
                ifd_corrected = ifd_data.dropna().copy()
                ifd_corrected['Restaurant_latitude'] = ifd_corrected['Restaurant_latitude'].abs()
                ifd_corrected['Restaurant_longitude'] = ifd_corrected['Restaurant_longitude'].abs()
                ifd_corrected['Delivery_location_latitude'] = ifd_corrected['Delivery_location_latitude'].abs()
                ifd_corrected['Delivery_location_longitude'] = ifd_corrected['Delivery_location_longitude'].abs()


                @st.cache_data
                def calculate_distance(df):
                    def dist_calc(row):
                        restaurant = (row['Restaurant_latitude'], row['Restaurant_longitude'])
                        delivery = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
                        return geodesic(restaurant, delivery).kilometers

                    df['distance_km'] = df.apply(dist_calc, axis=1)
                    return df


                ifd_corrected = calculate_distance(ifd_corrected)

            # ========== Overview ==========
            if model_section == "Overview":
                st.write("""
                To better capture the relationships in the data, we will add polynomial features and compare the performance with other models.

                We will look at 2 model comparisons:
                1. **Distance & Vehicle Condition and their interaction on Time**
                2. **Traffic & Weather and their Interactions on Time**

                Please select a model comparison from the radio buttons above.
                """)

            # ========== Model 1: Distance & Vehicle Condition ==========
            elif model_section == "Model 1: Distance & Vehicle Condition":
                st.write("## 🚗 Model Comparison 1: Distance & Vehicle Condition")
                st.write(
                    "Investigating the relationships using several regression models on key features (scaled distance, vehicle condition, and their interaction).")

                with st.spinner('Training models and evaluating performance...'):
                    scaler_zscore = StandardScaler()
                    ifd_corrected[['distance_km_zscore', 'vehicle_condition_zscore']] = scaler_zscore.fit_transform(
                        ifd_corrected[['distance_km', 'Vehicle_condition']]
                    )

                    feature_cols = ['distance_km_zscore', 'vehicle_condition_zscore']
                    X = ifd_corrected[feature_cols].copy()
                    X['interaction'] = X[feature_cols[0]] * X[feature_cols[1]]
                    y = ifd_corrected['Time_taken(min)']

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    poly_reg_model = Pipeline([
                        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                        ('lin_reg', LinearRegression())
                    ])

                    models = {
                        'Linear Regression': LinearRegression(),
                        'Polynomial Regression (d=2)': poly_reg_model,
                        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42,
                                                               n_jobs=-1),
                        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
                    }

                    results = []
                    for model_name, model in models.items():
                        model.fit(X_train, y_train)
                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)
                        train_r2 = r2_score(y_train, y_train_pred)
                        test_r2 = r2_score(y_test, y_test_pred)
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                        test_mae = mean_absolute_error(y_test, y_test_pred)
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

                        results.append({
                            'Model': model_name,
                            'Train_R²': train_r2,
                            'Test_R²': test_r2,
                            'Test_RMSE': test_rmse,
                            'Test_MAE': test_mae,
                            'CV_R²': cv_scores.mean(),
                            'Overfit': train_r2 - test_r2
                        })

                    results_df = pd.DataFrame(results)

                st.success('Model training complete!')

                st.dataframe(results_df.style.format({
                    'Train_R²': '{:.4f}',
                    'Test_R²': '{:.4f}',
                    'Test_RMSE': '{:.4f}',
                    'Test_MAE': '{:.4f}',
                    'CV_R²': '{:.4f}',
                    'Overfit': '{:.4f}'
                }))

                best_idx = results_df['Test_R²'].idxmax()
                best_model_name = results_df.loc[best_idx, 'Model']

                st.write(f"**Best Model:** {best_model_name} (Test R² = {results_df.loc[best_idx, 'Test_R²']:.4f})")

                # visuals
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go

                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'R² Score Comparison',
                        'Test Error Metrics',
                        'Overfitting Analysis',
                        f'Best Model: {best_model_name}'
                    ),
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )

                # subplot 1: R² Score Comparison
                fig.add_trace(
                    go.Bar(
                        x=results_df['Model'],
                        y=results_df['Train_R²'],
                        name='Train R²',
                        marker_color='rgb(55, 83, 109)',
                        text=results_df['Train_R²'].round(4),
                        textposition='auto'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Bar(
                        x=results_df['Model'],
                        y=results_df['Test_R²'],
                        name='Test R²',
                        marker_color='rgb(26, 118, 255)',
                        text=results_df['Test_R²'].round(4),
                        textposition='auto'
                    ),
                    row=1, col=1
                )

                # subplot 2: Test Error Metrics
                fig.add_trace(
                    go.Bar(
                        x=results_df['Model'],
                        y=results_df['Test_RMSE'],
                        name='RMSE',
                        marker_color='rgb(219, 64, 82)',
                        text=results_df['Test_RMSE'].round(2),
                        textposition='auto',
                        showlegend=False
                    ),
                    row=1, col=2
                )

                # subplot 3: Overfitting Analysis
                colors = ['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' for x in results_df['Overfit']]

                fig.add_trace(
                    go.Bar(
                        x=results_df['Model'],
                        y=results_df['Overfit'],
                        marker_color=colors,
                        text=results_df['Overfit'].round(4),
                        textposition='auto',
                        showlegend=False
                    ),
                    row=2, col=1
                )

                # subplot 4: Best Model Predictions
                best_model = models[best_model_name]
                y_pred = best_model.predict(X_test)

                fig.add_trace(
                    go.Scatter(
                        x=y_test,
                        y=y_pred,
                        mode='markers',
                        marker=dict(size=5, color='rgba(26, 118, 255, 0.5)'),
                        name='Predictions',
                        showlegend=False
                    ),
                    row=2, col=2
                )

                # perfect prediction line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        showlegend=False
                    ),
                    row=2, col=2
                )

                fig.update_xaxes(title_text="Model", row=1, col=1)
                fig.update_yaxes(title_text="R² Score", row=1, col=1)
                fig.update_xaxes(title_text="Model", row=1, col=2)
                fig.update_yaxes(title_text="Error (minutes)", row=1, col=2)
                fig.update_xaxes(title_text="Model", row=2, col=1)
                fig.update_yaxes(title_text="Train R² - Test R²", row=2, col=1)
                fig.update_xaxes(title_text="Actual Time (min)", row=2, col=2)
                fig.update_yaxes(title_text="Predicted Time (min)", row=2, col=2)

                fig.update_layout(
                    title_text='Model Performance Analysis for Distance & Vehicle Condition',
                    showlegend=True,
                    height=1200
                )

                st.plotly_chart(fig, use_container_width=True)

                # Polynomial Feature Coefficients
                st.write("### Polynomial Feature Coefficients (Feature Importance)")

                poly_reg_model.fit(X_train, y_train)
                coeffs = poly_reg_model.named_steps['lin_reg'].coef_
                poly_features = poly_reg_model.named_steps['poly'].get_feature_names_out(X.columns)

                coeff_df = pd.DataFrame({'feature': poly_features, 'coefficient': coeffs})

                fig_poly_coeffs = px.bar(
                    coeff_df,
                    x='coefficient',
                    y='feature',
                    orientation='h',
                    title='Polynomial Feature Coefficients'
                )
                st.plotly_chart(fig_poly_coeffs, use_container_width=True)

                most_important = coeff_df.loc[coeff_df['coefficient'].abs().idxmax()]

                st.markdown(f"""
                ### Conclusion
                The most important feature is **{most_important['feature']}** with a coefficient of **{most_important['coefficient']:.4f}**, 
                indicating it has the strongest impact on delivery time prediction.
                """)

            # ========== Model 2: Traffic & Weather ==========
            elif model_section == "Model 2: Traffic & Weather Interactions":
                st.write("## 🌦️ Model Comparison 2: Traffic & Weather Interactions")
                st.write(
                    "Investigating how **Road Traffic Density** and **Weather Conditions** affect delivery time, **including their interaction effects**.")

                st.write("""
                **Hypothesis:** Certain combinations of traffic and weather may have compounding effects. 
                For example, *Jam + Fog* might be worse than the sum of their individual effects.
                """)

                with st.spinner('Training models with traffic, weather, and their interactions...'):
                    # One-Hot Encoding
                    traffic_dummies = pd.get_dummies(ifd_corrected['Road_traffic_density'], prefix='traffic',
                                                     drop_first=True)
                    weather_dummies = pd.get_dummies(ifd_corrected['Weatherconditions'], prefix='weather',
                                                     drop_first=True)

                    # create correlation
                    interaction_features = pd.DataFrame()

                    for traffic_col in traffic_dummies.columns:
                        for weather_col in weather_dummies.columns:
                            interaction_name = f"{traffic_col}_X_{weather_col}"
                            interaction_features[interaction_name] = traffic_dummies[traffic_col] * weather_dummies[
                                weather_col]

                    # combine the features
                    X_tw_base = pd.concat([traffic_dummies, weather_dummies], axis=1)
                    X_tw_with_interaction = pd.concat([traffic_dummies, weather_dummies, interaction_features], axis=1)

                    y_tw = ifd_corrected['Time_taken(min)']

                    st.write(f"""
                    **Feature Engineering Summary:**
                    - Traffic categories: {traffic_dummies.shape[1]}
                    - Weather categories: {weather_dummies.shape[1]}
                    - Interaction terms: {interaction_features.shape[1]}
                    - **Total features (with interactions):** {X_tw_with_interaction.shape[1]}
                    - **Total features (without interactions):** {X_tw_base.shape[1]}
                    """)

                    # compare: with vs without features
                    results_comparison = []

                    for include_interaction, X_data, label in [
                        (False, X_tw_base, 'Without Interactions'),
                        (True, X_tw_with_interaction, 'With Interactions')
                    ]:

                        #rs = 42 if label == 'Without Interactions' else 123

                        X_train_tw, X_test_tw, y_train_tw, y_test_tw = train_test_split(
                            X_data, y_tw, test_size=0.2
                        )

                        for model_name in ['Linear Regression', 'Random Forest', 'Gradient Boosting']:
                            if model_name == 'Linear Regression':
                                model = LinearRegression()
                            elif model_name == 'Random Forest':
                                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=123,
                                                              n_jobs=-1)
                            elif model_name == 'Gradient Boosting':
                                model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=456)

                            st.write(f"Training {model_name} with {label}...")
                            st.write(f"Training data shape: {X_train_tw.shape}")

                            model.fit(X_train_tw, y_train_tw)

                            y_train_pred = model.predict(X_train_tw)
                            y_test_pred = model.predict(X_test_tw)
                            train_r2 = r2_score(y_train_tw, y_train_pred)
                            test_r2 = r2_score(y_test_tw, y_test_pred)
                            test_rmse = np.sqrt(mean_squared_error(y_test_tw, y_test_pred))
                            test_mae = mean_absolute_error(y_test_tw, y_test_pred)

                            results_comparison.append({
                                'Feature Set': label,
                                'Model': model_name,
                                'Train_R²': train_r2,
                                'Test_R²': test_r2,
                                'Test_RMSE': test_rmse,
                                'Test_MAE': test_mae,
                                'Overfit': train_r2 - test_r2
                            })

                    results_comparison_df = pd.DataFrame(results_comparison)

                st.success('Model training complete!')

                st.write("#### Model Performance: With vs Without Interactions")
                st.dataframe(results_comparison_df.style.format({
                    'Train_R²': '{:.4f}',
                    'Test_R²': '{:.4f}',
                    'Test_RMSE': '{:.4f}',
                    'Test_MAE': '{:.4f}',
                    'Overfit': '{:.4f}'
                }))

                # compare visualizations
                fig_comparison = px.bar(
                    results_comparison_df,
                    x='Model',
                    y='Test_R²',
                    color='Feature Set',
                    barmode='group',
                    title='Test R² Comparison: With vs Without Interaction Terms'
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

                st.write("### Interaction Effect Analysis")

                lr_with_interaction = LinearRegression()
                X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
                    X_tw_with_interaction, y_tw, test_size=0.2, random_state=42
                )
                lr_with_interaction.fit(X_train_full, y_train_full)

                coef_df = pd.DataFrame({
                    'Feature': X_tw_with_interaction.columns,
                    'Coefficient': lr_with_interaction.coef_
                })

                interaction_coefs = coef_df[coef_df['Feature'].str.contains('_X_')].copy()
                interaction_coefs['Abs_Coefficient'] = interaction_coefs['Coefficient'].abs()
                interaction_coefs = interaction_coefs.sort_values('Abs_Coefficient', ascending=False).head(10)

                st.write("#### Top 10 Strongest Interaction Effects (Linear Regression Coefficients)")

                fig_interactions = px.bar(
                    interaction_coefs,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title='Top 10 Traffic × Weather Interactions',
                    color='Coefficient',
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0
                )
                st.plotly_chart(fig_interactions, use_container_width=True)

                with st.expander("🔍 Click to see detailed interaction examples"):
                    st.write("""
                    **How to read the interaction coefficients:**

                    - **Positive coefficient (red):** This combination **increases** delivery time more than expected
                    - **Negative coefficient (blue):** This combination **decreases** delivery time more than expected

                    **Example:**
                    - If `traffic_Jam_X_weather_Fog` has coefficient = +5.2, it means:
                      - When BOTH Jam and Fog occur together, delivery time increases by an **extra** 5.2 minutes
                      - Beyond what you'd expect from just adding Jam's effect + Fog's effect separately

                    The interactions will show which **combinations** are particularly problematic or beneficial.
                    """)

                # Random Forest
                st.write("### Feature Importance (Random Forest)")

                rf_with_interaction = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                rf_with_interaction.fit(X_train_full, y_train_full)

                feature_importance_df = pd.DataFrame({
                    'Feature': X_tw_with_interaction.columns,
                    'Importance': rf_with_interaction.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)

                fig_rf_importance = px.bar(
                    feature_importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 15 Most Important Features (Random Forest)'
                )
                st.plotly_chart(fig_rf_importance, use_container_width=True)

                st.write("### Summary")

                best_tw_result = results_comparison_df.loc[results_comparison_df['Test_R²'].idxmax()]

                improvement = best_tw_result['Test_R²'] - results_comparison_df[
                    results_comparison_df['Feature Set'] == 'Without Interactions'
                    ]['Test_R²'].max()

                st.markdown(f"""
                **Key Insights:**

                - Without interactions: R² = {results_comparison_df[results_comparison_df['Feature Set'] == 'Without Interactions']['Test_R²'].max():.4f}
                - With interactions: R² = {best_tw_result['Test_R²']:.4f}
                - **Improvement from interactions: {improvement:.4f} ({improvement * 100:.1f}% of variance)**

                💡 **Conclusion:** 
                - Interaction effects {'only improve a little bit' if improvement > 0.02 else 'marginally improve' if improvement > 0.01 else 'minimally affect'} the model's predictive power.
                - The most impactful combinations are shown in the interaction coefficient chart above.
                - **Most Important factor for determining time: 1. Traffic Low 2. Sunny Weather** 
                """)

            # ========== Overall Conclusion ==========
            # ========== Overall Conclusion ==========
            elif model_section == "Conclusion: What Affects Delivery Time Most?":
                st.write("## 🎯 Conclusion: What Affects Delivery Time Most?")
                st.write("Based on our comprehensive analysis across all models and features:")

                with st.spinner('Calculating effect sizes...'):
                    ifd_corrected_temp = ifd_data.dropna().copy()
                    ifd_corrected_temp['Restaurant_latitude'] = ifd_corrected_temp['Restaurant_latitude'].abs()
                    ifd_corrected_temp['Restaurant_longitude'] = ifd_corrected_temp['Restaurant_longitude'].abs()
                    ifd_corrected_temp['Delivery_location_latitude'] = ifd_corrected_temp[
                        'Delivery_location_latitude'].abs()
                    ifd_corrected_temp['Delivery_location_longitude'] = ifd_corrected_temp[
                        'Delivery_location_longitude'].abs()


                    def calculate_distance_temp(df):
                        def dist_calc(row):
                            restaurant = (row['Restaurant_latitude'], row['Restaurant_longitude'])
                            delivery = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
                            return geodesic(restaurant, delivery).kilometers

                        df['distance_km'] = df.apply(dist_calc, axis=1)
                        return df


                    ifd_corrected_temp = calculate_distance_temp(ifd_corrected_temp)

                    def calculate_eta_squared(df, categorical_var, continuous_var):
                        """single factor eta-squared"""
                        groups = df.groupby(categorical_var)[continuous_var].apply(list)
                        f_stat, p_value = stats.f_oneway(*groups)

                        # eta-squared
                        grand_mean = df[continuous_var].mean()
                        ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)
                        ss_total = sum((x - grand_mean) ** 2 for x in df[continuous_var])
                        eta_squared = ss_between / ss_total

                        return eta_squared, f_stat, p_value


                    # calculate all value η²
                    categorical_vars = {
                        'Road Traffic Density': 'Road_traffic_density',
                        'Weather Conditions': 'Weatherconditions',
                        'Vehicle Type': 'Type_of_vehicle',
                        'Vehicle Condition': 'Vehicle_condition',
                        'Festival': 'Festival',
                        'City': 'City'
                    }

                    categorical_results = []
                    for name, var in categorical_vars.items():
                        if var in ifd_corrected_temp.columns:
                            try:
                                eta_sq, f_stat, p_val = calculate_eta_squared(
                                    ifd_corrected_temp, var, 'Time_taken(min)'
                                )
                                categorical_results.append({
                                    'Factor': name,
                                    'η²': eta_sq,
                                    'F-statistic': f_stat,
                                    'p-value': p_val
                                })
                            except:
                                pass

                    categorical_df = pd.DataFrame(categorical_results).sort_values('η²', ascending=False)


                    def calculate_r_squared(df, predictor, target):
                        """simple linear regression R²"""
                        X = df[[predictor]].values
                        y = df[target].values
                        slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
                        return r_value ** 2, p_value


                    continuous_vars = {
                        'Distance (km)': 'distance_km',
                        'Vehicle Condition (numeric)': 'Vehicle_condition',
                        'Delivery Person Age': 'Delivery_person_Age',
                        'Delivery Person Ratings': 'Delivery_person_Ratings'
                    }

                    continuous_results = []
                    for name, var in continuous_vars.items():
                        if var in ifd_corrected_temp.columns:
                            try:
                                r_sq, p_val = calculate_r_squared(
                                    ifd_corrected_temp, var, 'Time_taken(min)'
                                )
                                continuous_results.append({
                                    'Factor': name,
                                    'R²': r_sq,
                                    'p-value': p_val
                                })
                            except:
                                pass

                    continuous_single_df = pd.DataFrame(continuous_results).sort_values('R²', ascending=False)

                st.markdown("### 📊 Single Factor Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("#### Categorical Variables (η²)")
                    st.dataframe(categorical_df.style.format({
                        'η²': '{:.4f}',
                        'F-statistic': '{:.2f}',
                        'p-value': '{:.2e}'
                    }))

                    st.info("""
                    **η² Interpretation:**
                    - 0.01 = Small effect
                    - 0.06 = Medium effect
                    - 0.14+ = Large effect
                    """)

                with col2:
                    st.write("#### Continuous Variables (R²)")
                    st.dataframe(continuous_single_df.style.format({
                        'R²': '{:.4f}',
                        'p-value': '{:.2e}'
                    }))

                    st.info("""
                    **R² Interpretation:**
                    - Variance explained by single variable
                    - Higher = stronger predictor
                    """)

                # Multi-variable models
                st.markdown("### 🔬 Multi-Variable Models (R²)")

                multi_var_models = pd.DataFrame({
                    'Model': [
                        'Distance + Vehicle Condition + Interaction',
                        'Distance alone (Linear Regression)'
                    ],
                    'R²': [0.59, 0.55],
                    'Number of Variables': [3, 1]
                }).sort_values('R²', ascending=False)

                st.dataframe(multi_var_models.style.format({'R²': '{:.4f}'}))

                # Key findings
                st.markdown("### 🏆 Key Findings")

                top_categorical = categorical_df.iloc[0]
                top_continuous = continuous_single_df.iloc[0]

                st.success(f"""
                **Most Important Single Factors:**
                - **Categorical:** {top_categorical['Factor']} (η² = {top_categorical['η²']:.4f}, explains {top_categorical['η²'] * 100:.2f}% of variance)
                - **Continuous:** {top_continuous['Factor']} (R² = {top_continuous['R²']:.4f}, explains {top_continuous['R²'] * 100:.2f}% of variance)
                - **Best Multi-Variable Model:** Distance + Vehicle Condition (R² = 0.59, explains 59% of variance)
                """)

                st.markdown("""
                ### 💡 Conclusion

                From this Analysis, we are surprised to find that distance is not the biggest contributor towards the delivery time. We can actually see that it was Road Traffic density that matters more.
                
                In terms of effect, Distance does not have a lot to do with Vehicle condition. With a poorer or better vehicle condition, same distance, it does not increase to decrease the delivery time by too much.
                
                Delivery person's rating was a shock for me, as I originally neglected this column as I assume it was not instrumental. It turned out to have a more pronounced effect than Distance! 
                
                Traffic on the other hand, with low traffic and sunny weather, it can reduce the delivery time for almost 30% compared to normal.
                
                This dataset suggests a lot more factors affecting delivery time than the previous 2 datasets.
                
                """)

elif dataset_choice == "Overall Conclusion and Findings":
    st.header("🎯 Overall Conclusion and Findings")

    st.markdown("""
    ## Summary of Factors Affecting Delivery Time

    These are factors that we've discovered across 3 datasets:
    1. **Distance**
    2. **Traffic Condition** 
    3. **Delivery Ratings** 
    4. **Weather**

    
    
    Below are a few thing that we've found that has little (<5%) or no impact on delivery time:
    1. Restaurant Ratings
    2. Cost of Order
    3. Food Preparation Time
    4. Courier Experiences
    5. Vehicle Condition
    6. Vehicle Type
    """)