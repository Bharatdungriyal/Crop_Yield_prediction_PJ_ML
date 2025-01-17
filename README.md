# Crop Yield Prediction Using Supervised Machine Learning

Welcome to the Crop Yield Prediction project! This project leverages supervised machine learning techniques to predict the yield of crops based on various factors. The project is built with Python and includes a web interface for user input, using Bootstrap for styling.

## Project Overview

### **Features**

- **Data Processing**: Includes handling missing values, encoding categorical variables, and scaling numerical features.
- **Machine Learning**: Utilizes various supervised learning algorithms to predict crop yield.
- **Web Interface**: Built with Bootstrap for a responsive and user-friendly experience.

## Dependencies
Python

NumPy

Pandas

Seaborn

Matplotlib

scikit-learn

Flask

Bootstrap

## Installation
Clone the repository:

bash
git clone https://github.com/Bharatdungriyal/Crop_Yield_prediction_PJ_ML
Install the required dependencies:

bash
pip install -r requirements.txt
Usage
Run the web application:

bash
python app.py
Access the web interface: Open your browser and go to http://127.0.0.1:5000.

Input the necessary parameters: Use the web form to input parameters such as Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, and Item.

View the predictions: The application will display the predicted crop yield based on the provided input.

Project Structure
data/: Datasets for training and testing obtained from Kaggle.

models/: Trained machine learning models.

notebooks/: Jupyter notebooks for exploratory data analysis and model training.

app/: The Flask application for the web interface.

static/: Static files (CSS, JS, and images) for the web interface created using Bootstrap.

templates/: HTML templates for the web interface.

Data Processing
Libraries
python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
Encoding
Categorical variables are converted to numerical values using OneHotEncoder.

Scaling
Numerical features are standardized using StandardScaler.

Model Training
Libraries
python
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
Models Tested
The following supervised learning algorithms are tested:

LinearRegression()

Lasso()

Ridge()

KNeighborsRegressor()

DecisionTreeRegressor()

Evaluation Metrics
Mean Squared Error (MSE)

R-squared

Selected Model
The models are trained using the DecisionTreeRegressor, which showed the best performance based on MSE and R-squared metrics.
