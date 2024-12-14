import streamlit as st
import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, ShuffleSplit 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import AdaBoostRegressor
import joblib

# Load CSS styling
st.markdown(
    """
    <style>
    .box {
        background-color: #f0f2f6;
        padding: 5px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Sidebar selection
option = st.sidebar.radio("Regression Model", ['Sampling Technique/s', 'Performance Metric/s','Hyperparameter Tuning'])

# Regression Model
if option == 'Sampling Technique/s':
    st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        color: #4CAF50; /* Green color */
        text-align: center;
        font-family: 'Arial', sans-serif;
        margin-top: -70px;
        margin-bottom: 10px;
    }
     </style>
     """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Water Quality Prediction Model</h1>', unsafe_allow_html=True)

   
    # Function to load the dataset
    @st.cache_data
    def load_water_data(uploaded_file):
        names = ['aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine',
                'chromium', 'copper', 'fluoride', 'bacteria', 'viruses', 'lead',
                'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium',
                'selenium', 'silver', 'uranium', 'is_safe']
        dataframe = pd.read_csv(uploaded_file, names=names)
        
        dataframe.replace('#NUM!', pd.NA, inplace=True)
        dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

        # Impute missing values for numeric data (mean strategy)
        numeric_imputer = SimpleImputer(strategy='mean')
        dataframe.iloc[:, :-1] = numeric_imputer.fit_transform(dataframe.iloc[:, :-1])

        # Encode target variable 'is_safe' if necessary (assuming it's categorical)
        if dataframe['is_safe'].dtype == 'object' or dataframe['is_safe'].isnull().any():
            label_encoder = LabelEncoder()
            dataframe['is_safe'] = label_encoder.fit_transform(dataframe['is_safe'].fillna(0))

        return dataframe
    st.write("<br><br>", unsafe_allow_html=True)

    def get_water_quality_input():
        aluminium = st.number_input("Aluminium", min_value=0.000, value=0.000, key="aluminium1")
        ammonia = st.number_input("Ammonia", min_value=0.000, value=0.000, key="ammonia1")
        arsenic = st.number_input("Arsenic", min_value=0.000, value=0.000, key="arsenic1")
        barium = st.number_input("Barium", min_value=0.000, value=0.000, key="barium1")
        cadmium = st.number_input("Cadmium", min_value=0.000, value=0.000, key="cadmium1")
        chloramine = st.number_input("Chloramine", min_value=0.000, value=0.000, key="chloramine1")
        chromium = st.number_input("Chromium", min_value=0.000, value=0.000, key="chromium1")
        copper = st.number_input("Copper", min_value=0.000, value=0.000, key="copper1")
        fluoride = st.number_input("Fluoride", min_value=0.000, value=0.000, key="fluoride1")
        bacteria = st.number_input("Bacteria", min_value=0.000, value=0.000, key="bacteria1")
        viruses = st.number_input("Viruses", min_value=0.000, value=0.000, key="viruses1")
        lead = st.number_input("Lead", min_value=0.000, value=0.000, key="lead1")
        nitrates = st.number_input("Nitrates", min_value=0.000, value=0.000, key="nitrates1")
        nitrites = st.number_input("Nitrites", min_value=0.000, value=0.000, key="nitrites1")
        mercury = st.number_input("Mercury", min_value=0.000, value=0.000, key="mercury1")
        perchlorate = st.number_input("Perchlorate", min_value=0.000, value=0.000, key="perchlorate1")
        radium = st.number_input("Radium", min_value=0.000, value=0.000, key="radium1")
        selenium = st.number_input("Selenium", min_value=0.000, value=0.000, key="selenium1")
        silver = st.number_input("Silver", min_value=0.000, value=0.000, key="silver1")
        uranium = st.number_input("Uranium", min_value=0.000, value=0.000, key="uranium1")

        features = np.array([[aluminium, ammonia, arsenic, barium, cadmium, chloramine,
                            chromium, copper, fluoride, bacteria, viruses, lead,
                            nitrates, nitrites, mercury, perchlorate, radium,
                            selenium, silver, uranium]])
        return features

    def main():
        #Split into Train and Test Sets
         st.markdown("""
    <style>
    .split {
        font-size: 25px;
        color: black; /* Green color */
        font-family: 'Arial', sans-serif;
        margin-top: -70px;
        margin-bottom: 10px;
    }
     </style>
     """, unsafe_allow_html=True)
    st.markdown('<h1 class="split">Split into Train and Test Sets</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv", key="file_uploader_unique_key")

    if uploaded_file is not None:
            # Read dataset
            dataframe = load_water_data(uploaded_file)

            # Normalize column names
            dataframe.columns = dataframe.columns.str.strip().str.lower()  # Lowercase and strip whitespace

            # Debugging: Check the shape and columns of the dataset
            st.write("### Shape of the Dataset:")
            st.write(dataframe.shape)  # Show number of rows and columns

            st.write("### Dataset Preview:")
            st.write(dataframe.head())  # Show the first few rows of the dataset

            st.write("### Dataset Columns:")
            st.write(dataframe.columns.tolist())  # Show column names as a list

            st.write("### Check for Missing Values:")
            st.write(dataframe.isnull().sum())
            st.write("### Correlation Matrix:")
            st.write(dataframe.corr())

            # Check for any non-numeric columns
            non_numeric_cols = dataframe.select_dtypes(include=['object']).columns
            if len(non_numeric_cols) > 0:
                st.write(f"Non-numeric columns found: {non_numeric_cols}. Please ensure only numeric data is used for model training.")

            array = dataframe.values
            X = array[:, 0:20]  
            Y = array[:, 20]   

            st.write("### Shape of the Dataset:")
            st.write(dataframe.shape)  # This will show you how many rows and columns are in your dataset

            # Split into input and output variables
            X = dataframe.drop('is_safe', axis=1).values
            Y = dataframe['is_safe'].values

            test_size = st.slider("Test size (as a percentage)", 10, 50, 20) / 100
            seed = 7

            # Split the dataset into test and train
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

            # Train the data on a Logistic Regression model
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, Y_train)

            # Evaluate the accuracy
            result = model.score(X_test, Y_test)
            accuracy = result * 100
            st.write(f"Accuracy: {accuracy:.3f}%")

            # Interpretation of the accuracy
            if accuracy < 0:
                st.warning("The model's performance is poor. Please check your data for quality and relevance.")
            elif accuracy < 50:
                st.warning("The model has low predictive power. Consider gathering more data or using different features.")
            elif accuracy < 75:
                st.success("The model has moderate predictive power. It may perform reasonably well.")
            else:
                st.success("The model has high predictive power.")

            # Save and download the model
            model.fit(X, Y)
            model_filename = "logistic_regression_split_regression.joblib"
            joblib.dump(model, model_filename)
            st.success(f"Model saved as {model_filename}")

            # Option to download the model
            with open(model_filename, "rb") as f:
                st.download_button("Download Model", f, file_name=model_filename)

            # Model upload for prediction
            st.subheader("Upload a Saved Model for Prediction")
            uploaded_model = st.file_uploader("Upload your model (joblib format)", type=["joblib"])

            if uploaded_model is not None:
                model = joblib.load(uploaded_model)  # Load the uploaded model

                # Input fields for prediction
                st.subheader("Input Sample Data for Water Quality Prediction")
                input_data = get_water_quality_input()
                
                if st.button("Predict"):
                    # Prediction
                    prediction = model.predict(input_data)
                    
                    # Display prediction result
                    st.subheader("Prediction Result and Interpretation")
                    st.write(f"The predicted water safety is: **{'Safe' if prediction[0] == 1 else 'Not Safe'}**")
                    interpretation = "The water is safe for human consumption." if prediction[0] == 1 else "The water is not safe for human consumption."
                    st.write(interpretation)

    if __name__ == "__main__":
        main()  

# Performance Metric
elif option == 'Performance Metric/s':
    st.markdown("""
    <style>
    .perf {
        font-size: 25px;
        color: black;
        text-align: center;
        font-family: 'Arial', sans-serif;
        margin-top: -70px;
        margin-bottom: 10px;
    }
     </style>
     """, unsafe_allow_html=True)
    st.markdown('<h1 class="perf">PERFORMANCE METRICS</h1>', unsafe_allow_html=True)
    
    # Introduction and Instructions
    st.write("""       
        ### Instructions:
        1. Upload your CSV dataset.
        2. Click the button to evaluate the model.
        """)
    st.write("<br><br>", unsafe_allow_html=True)

    @st.cache_data
    def load_data(uploaded_file):
        names = ['aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine',
                'chromium', 'copper', 'fluoride', 'bacteria', 'viruses', 'lead',
                'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium',
                'selenium', 'silver', 'uranium', 'is_safe']
        dataframe = pd.read_csv(uploaded_file, names=names)
        
        dataframe.replace('#NUM!', pd.NA, inplace=True)
        dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

        # Impute missing values for numeric data (mean strategy)
        numeric_imputer = SimpleImputer(strategy='mean')
        dataframe.iloc[:, :-1] = numeric_imputer.fit_transform(dataframe.iloc[:, :-1])

        # Encode target variable 'is_safe' if necessary (assuming it's categorical)
        if dataframe['is_safe'].dtype == 'object' or dataframe['is_safe'].isnull().any():
            label_encoder = LabelEncoder()
            dataframe['is_safe'] = label_encoder.fit_transform(dataframe['is_safe'].fillna(0))

        return dataframe
    

    def main():
            st.markdown("""
        <style>
        .mean {
        font-size: 25px;
        color: black;
        font-family: 'Arial', sans-serif;
        margin-top: -70px;
        margin-bottom: 10px;
        }
        </style>
         """, unsafe_allow_html=True)
    st.markdown('<h1 class="mean">Mean Squared Error (MSE) (80:20 train-test split)</h1>', unsafe_allow_html=True)

        # Unique file uploader for CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="unique_file_uploader2")

    if uploaded_file is not None:
            # Load dataset
            st.write("Loading the dataset...")
            dataframe = load_data(uploaded_file)

            # Display first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Preparing the data
            array = dataframe.values
            X = array[:, :-1]  # Features (all columns except the last one)
            Y = array[:, -1]   # Target variable (last column)

            # Split the dataset into an 80:20 train-test split
            test_size = 0.2
            seed = 42  # Random seed for reproducibility
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

            # Train the data on a Linear Regression model
            model = LinearRegression()
            model.fit(X_train, Y_train)

            # Make predictions on the test data
            Y_pred = model.predict(X_test)

            # Calculate the mean squared error on the test set
            mse = mean_squared_error(Y_test, Y_pred)

            # Display results
            st.subheader("Model Evaluation")
            st.write(f"Mean Squared Error (MSE): {mse:.3f}")

            # Interpretation and guidance
            st.subheader("Interpretation of Results")
            st.write(
                """
                The Mean Squared Error (MSE) is a measure of how well the linear regression model predicts the target variable. 
                Specifically, MSE represents the average squared difference between the predicted values and the actual values. 
                A lower MSE indicates a better fit of the model to the data, meaning that the predictions are closer to the actual outcomes.

                Here's a general guideline for interpreting the MSE value:
                - **MSE = 0:** Perfect model with no prediction error.
                - **0 < MSE < 10:** Good model performance. The predictions are reasonably close to the actual values.
                - **10 ≤ MSE < 100:** Moderate model performance. The predictions have some error and may require further improvement.
                - **MSE ≥ 100:** Poor model performance. The predictions are far from the actual values, indicating that the model may not be capturing the underlying trends in the data well.
                """
            )
    else:
            st.write("Please upload a CSV file to proceed.")
    st.write("<br><br>", unsafe_allow_html=True)

    if __name__ == "__main__":
        main()
    
    def main():
        st.markdown("""
     <style>
        .abs {
        font-size: 25px;
        color: black;
        font-family: 'Arial', sans-serif;
        margin-top: -70px;
        margin-bottom: 10px;
        }
        </style>
         """, unsafe_allow_html=True)
    st.markdown('<h1 class="abs">Mean Absolute Error (MAE) (K-fold Cross Validation)</h1>', unsafe_allow_html=True)

        # Unique file uploader for CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="unique_file_uploader")

    if uploaded_file is not None:
            # Load dataset
            st.write("Loading the dataset...")
            dataframe = load_data(uploaded_file)

            # Display first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Preparing the data
            array = dataframe.values
            X = array[:, :-1]  # Features
            Y = array[:, -1]   # Target variable

            # Set up cross-validation
            kfold = KFold(n_splits=10, random_state=None)
            
            # Train the model
            model = LinearRegression()

            # Calculate the mean absolute error
            scoring = 'neg_mean_absolute_error'
            results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            
            # Display results
            st.subheader("Cross-Validation Results")
            mae = -results.mean()
            std_dev = results.std()
            st.write(f"Mean Absolute Error (MAE): {mae:.3f} (+/- {std_dev:.3f})")

            # Interpretation and guidance
            st.subheader("Interpretation of Results")
            st.write(
                """
                The Mean Absolute Error (MAE) is a measure of prediction accuracy for regression models. 
                It represents the average absolute difference between the predicted values and the actual values. 
                A lower MAE indicates better model performance, meaning the predictions are closer to the actual outcomes.

                Here's how to interpret the MAE value:
                - **MAE = 0:** Perfect model with no prediction error.
                - **0 < MAE < 10:** Good model performance, with predictions reasonably close to the actual values.
                - **10 ≤ MAE < 50:** Moderate model performance; the predictions have some error and may need improvement.
                - **MAE ≥ 50:** Poor model performance; the predictions are far from the actual values, indicating that the model may not capture the underlying trends well. 
                """
            )
    else:
            st.write("Please upload a CSV file to proceed.")
    st.write("<br><br>", unsafe_allow_html=True)    

    if __name__ == "__main__":
        main()    

####################################################################################################
####################################################################################################

# Hyperparameter Tuning
elif option == 'Hyperparameter Tuning':
    st.title("HYPERPARAMETER TUNING")

    # Hyperparameter Tuning for AdaBoost
    st.header("AdaBoost Regressor")

    def main():
        # Load the dataset
        filename = r'C:\Users\admin\Documents\ITD112\lab3\water.csv'
        dataframe = pd.read_csv(filename)

        # Ensure all columns are numeric, and handle non-numeric entries by replacing them with NaN
        dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values
        dataframe = dataframe.dropna()

        # Separate features and target
        X = dataframe.iloc[:, :-1].values
        Y = dataframe.iloc[:, -1].values

        # Set Hyperparameters
        st.subheader("Set Hyperparameters")
        n_estimators = st.slider("Number of Estimators", 1, 200, 50, 1)
        learning_rate = st.slider("Learning Rate", 0.01, 5.0, 1.0, 0.01)

        # K-Fold Cross-Validation
        kfold = KFold(n_splits=10, random_state=42, shuffle=True)

        # Initialize AdaBoost Model
        ada_model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

        # Cross-Validation for Mean Absolute Error
        scoring = 'neg_mean_absolute_error'
        ada_results = cross_val_score(ada_model, X, Y, cv=kfold, scoring=scoring)

        # Calculate and Display Results
        mean_mae = -ada_results.mean()
        std_dev_mae = ada_results.std()
        
        st.write("<br>", unsafe_allow_html=True)
        st.write(f"### Results for AdaBoost Regressor")
        st.write(f"- **Number of Estimators**: {n_estimators}")
        st.write(f"- **Learning Rate**: {learning_rate}")
        st.write(f"- **Mean Absolute Error (MAE)**: {mean_mae:.3f}")
        st.write(f"- **Standard Deviation of MAE**: {std_dev_mae:.3f}")
        st.write("Completed tuning for AdaBoost Regressor.")
        st.write("<br><br>", unsafe_allow_html=True)

        
        

    # Call main function to run code in Streamlit
    if __name__ == "__main__":
        main()
