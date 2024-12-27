This repository contains a project focused on predicting housing prices using various regression models. 
The application analyzes key features of homes and provides insights into their impact on price predictions, leveraging a real-world dataset from Kaggle.

Key Features:
Data Transformation:
Standardized features for consistent model input using StandardScaler.
Engineered and selected relevant features like bathrooms, bedrooms, waterfront, view, sqft_living, and yr_built.

Dynamic Model Implementations:
Linear Regression:
Fitted to understand linear relationships between features and prices.
K-Nearest Neighbors (KNN) Regression:
Predicts housing prices by evaluating the influence of neighboring data points.
Decision Tree Regression:
Constructs tree-based models for non-linear relationships.
Random Forest Regression:
Uses an ensemble of decision trees for robust price prediction and feature importance analysis.

Visualizations:
Correlation Heatmap:
Displays relationships between housing features and price.
Performance Metrics:
Includes R² scores and cross-validated R² scores for model evaluation.

Interactive Analysis:
Prints detailed comparisons of actual vs. predicted housing prices.
Outputs feature importance for better model interpretability.

Technologies Used
Python
pandas, numpy, scikit-learn for data manipulation and modeling.
matplotlib, seaborn for visualizations.
