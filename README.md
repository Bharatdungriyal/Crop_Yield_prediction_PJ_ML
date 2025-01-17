# Crop Yield Prediction Using Supervised Machine Learning

Welcome to the Crop Yield Prediction project! This project leverages supervised machine learning techniques to predict the yield of crops based on various factors. The project is built with Python and includes a web interface for user input, using Bootstrap for styling.

## Project Overview

### **Features**

- **Data Processing**: Includes handling missing values, encoding categorical variables, and scaling numerical features.
- **Machine Learning**: Utilizes various supervised learning algorithms to predict crop yield.
- **Web Interface**: Built with Bootstrap for a responsive and user-friendly experience.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crop-yield-prediction.git
   cd crop-yield-prediction
2. Install the required dependencies:
bash
pip install -r requirements.txt
Usage
Run the web application:

bash
python app.py
Access the web interface: Open your browser and go to http://localhost:5000.

Input the necessary parameters: Use the web form to input parameters such as Year,	average_rain_fall_mm_per_year,	pesticides_tonnes,	avg_temp,	Area,	Item.

View the predictions: The application will display the predicted crop yield based on the provided input.

Project Structure
data/: got the datasets used for training and testing from kaggle.

models/: the trained machine learning models.

notebooks/: Jupyter notebooks for exploratory data analysis and model training.

app/: The Flask application for the web interface, using flask docs

static/: Contains static files (CSS, JS, images) for the web interface.

templates/: Contains HTML templates for the web interface.

Data Processing
python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
Encoding: Categorical variables are converted to numerical using OneHotEncoder.

Scaling: Numerical features are standardized using StandardScaler.

Model Training
The models are tested using various supervised learning algorithms such as LinearRegression(),
Lasso(),
Ridge(),
KNeighborsRegressor(),
DecisionTreeRegressor(). 
The performance of each model is evaluated using metrics like Mean Squared Error (MSE) and R-squared.
Model Training
The models are trained using the Decision Tree Regressor, which showed the best performance based on Mean Squared Error (MSE) and R-squared metrics.
