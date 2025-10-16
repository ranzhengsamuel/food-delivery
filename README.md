**Food Delivery Time Analysis & Prediction**

This repository presents a data analytics and prediction project on food delivery times, using two real-world datasets:

NYC_food_order.csv
Food_Delivery_Times.csv

The goal is to understand what really affects delivery time, from so many factors, pick the most influential factor that affects the delivery time and if possible, analyze on how much the influence is.

The repository includes:

Data preprocessing and cleaning scripts / notebooks
Exploratory Data Analysis (EDA) with visualizations (e.g. correlation heatmaps)
Model presentation and evaluation
A Streamlit app to present predictions interactively
A dashboard / interactive visualizations

**Datasets**

NYC_food_order.csv — contains records of food orders in New York City, with fields such as order timestamp, delivery location, etc.
Food_Delivery_Times.csv — contains timestamps (order time, pick-up, delivery), distances, and possibly other features related to delivery process.

These datasets were selected because they are sufficiently large and rich to allow meaningful modeling. (As stated in the repo, the datasets “have a large dataset with enough entry, and the items are comprehensive.”)

**Preprocessing & Imputation**

Some data points had missing values. Imputation methods were used to fill in these gaps.
A key insight is that missing data did not materially alter the correlation structure (i.e. before imputation correlation was weak, and after imputation still weak) 
GitHub

Standard cleaning steps such as:

~Type conversions (dates, numeric)

~Removing duplicates or erroneous rows

~Factoring out the items that are not necesary to analyze (for example, customer ID and order ID)

~Normalization / scaling if needed

**Questions & Hypotheses Addressed (Research Questions)**

1. Which features most influence delivery time?
Initial expectation: distance, traffic, preparation time, courier experience, weather, etc.

2. Is distance the strongest predictor of traveling time?
From common sense perspective, it does seem that distance is a big indicator of traveling time, I would like to see whether that is the case for this dataset.

3. How do other factors (traffic, time of day, weather, courier experience) modulate delivery times?
What level of correlation they have against traveling time?

**Analysis & Key Findings**

From the exploratory and modeling work, the following observations and conclusions emerged:

~ Distance is the strongest predictor — it consistently exhibits the strongest correlation with delivery time; other features (traffic, weather, courier experience) have weak or inconsistent correlations.

~ Limited marginal value of other features — after controlling for distance, additional variables often add little explanatory power.

~ Interactive dashboard — the Streamlit app enables users to test “what-if” scenarios: e.g. varying distance, time of day, or hypothetical feature adjustments, and see predicted delivery times.

~ Implications: In operational settings, focusing on reducing distance (e.g. assigning orders to closer couriers, optimizing routing) may yield the most significant gains. Other interventions (traffic-aware routing, improved courier training) may yield incremental benefits.

**Conclusion**

This project demonstrates that, at least for the datasets analyzed, distance plays an overwhelmingly dominant role in predicting food delivery time. While other factors (traffic congestion, time of day, courier experience, weather) might intuitively matter, their marginal explanatory power is limited in this context.

Hence, any practical system for estimating or optimizing delivery times should prioritize distance-based heuristics and routing logic. More sophisticated models that incorporate many additional features may yield diminishing returns, depending on the quality and resolution of the data.
