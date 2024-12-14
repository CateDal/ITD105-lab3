from itertools import islice
import time
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Perceptron, Lasso, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import joblib
import io
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch



st.sidebar.markdown(
    "<h2 style='text-align: center; font-weight: bold; color: #333; margin-bottom: -40px; margin-top: -20px;'>GlucoSense</h2>",
    unsafe_allow_html=True,
)


# Step navigation
steps = ["Model Training and Comparison", "Model Usage"]
step = st.sidebar.radio("", steps)

# Session state for carrying data between steps
if "data" not in st.session_state:
    st.session_state.data = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

# Step 1: Upload Data and Step 2: Model Training
if step == "Model Training and Comparison":
    st.subheader("Step 1: Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        
        tabs = st.tabs(["Preview of the Dataset", "Whole Dataset", "Shape of the Dataset","Summary Statistics"])
        with tabs[0]:
            st.write("Preview of the dataset:")
            st.dataframe(data.head())
        with tabs[1]:
            st.write("Number of Data (row,column)", data.shape)
            
            st.write("Dataset:")
            st.write(data)
        with tabs[2]:
           # Create two columns for Data Info and Missing Values
            col1, col2 = st.columns(2)

            # Data Info in the first column
            with col1:
                buffer = io.StringIO()
                data.info(buf=buffer)
                s = buffer.getvalue()
                info_lines = s.split('\n')[5:-2]  # Extract lines containing column info
                table_data = []

                # Parse each line to extract details
                for line in info_lines:
                    parts = line.split()
                    col_name = parts[0]
                    non_null_count = parts[1]
                    dtype = parts[-1]
                    table_data.append([col_name, non_null_count, dtype])

                # Create a DataFrame for display
                info_df = pd.DataFrame(table_data, columns=["Column Name", "Non-Null Count", "Data Type"])

                # Display in Streamlit
                st.write("Data Info Table")
                st.dataframe(info_df)
                

            # Missing Values in the second column
            with col2:
                st.write('Missing Values')
                st.write(data.isnull().sum())
                df = data.fillna(data.select_dtypes(include=[float, int]).mean())
            
        with tabs[3]:    
            st.subheader('Summary Statistics')
            st.write(data.describe())

                
    else:
        st.warning("Please upload a CSV file to proceed.")
    
    st.write("<br><br>", unsafe_allow_html=True)
    st.subheader("Step 2: Model Training and Comparison")

    if st.session_state.data is None:
        st.warning("Please upload a dataset first in Step 1.")
    else:
        data = st.session_state.data
        st.success("Dataset loaded successfully!")

        # Select mode: Classification or Regression
        st.markdown(
        "<h3 style='font-size: 23px; margin-top:25px;'>Classification</h3>", 
        unsafe_allow_html=True
            )

        target = st.selectbox("Select the target variable:", data.columns)
        features = st.multiselect("Select feature variables:", data.columns, default=[col for col in data.columns if col != target])

        if features and target:
            X = data[features]
            y = data[target]

            # Handle missing values
            st.markdown(
            "<h3 style='font-size: 20px;'>Handle Missing Data</h3>", 
            unsafe_allow_html=True
            )
            handle_missing = st.radio(
                "Choose how to handle missing values:",
                ["Impute with Mean", "Drop Rows with Missing Values"]
            )
            if handle_missing == "Impute with Mean":
                X.fillna(X.mean(), inplace=True)
                y.fillna(y.mean(), inplace=True)
            elif handle_missing == "Drop Rows with Missing Values":
                X.dropna(inplace=True)
                y = y.loc[X.index]  # Ensure the target matches the filtered features

           # Split data. Set the test size using a slider
            test_size = st.slider("Test Size", 0.1, 0.8, 0.2, key="test_size")
            random_seed = st.slider("Random Seed", 1, 100, 7, key="random_seed")

            # Perform the train-test split using the slider values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

            # Save the splits to session state for further use
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.session_state.y_train, st.session_state.y_test = y_train, y_test

            st.success("Data split into training and test sets!")
            st.write(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

            models = {
                "Decision Tree (CART)": DecisionTreeClassifier(),
                "Naive Bayes": GaussianNB(),
                "Gradient Boosting Machines (AdaBoost)": AdaBoostClassifier(),
                "K-Nearest Neighbors (K-NN)": KNeighborsClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Multi-Layer Perceptron (MLP)": MLPClassifier(),
                "Perceptron": Perceptron(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector Machines (SVM)": SVC(probability=True),
            }
            metric_fn = accuracy_score
            metric_name = "Accuracy"
            # Train models
            if st.button("Train Models"):
               # X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

                results = []
                total_models = len(models)
                progress_bar = st.progress(0)
                progress_text = st.empty()

                for idx, (name, model) in enumerate(models.items()):
                    progress_text.text(f"Training {name} ({idx + 1}/{total_models})...")

                    # Train the model
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metric_value = metric_fn(y_test, y_pred)
                    elapsed_time = time.time() - start_time

                    results.append({
                        "Model": name,
                        metric_name: metric_value,
                        "Time (s)": elapsed_time,
                    })

                    progress_bar.progress((idx + 1) / total_models)

                progress_text.text("Training complete!")
                results_df = pd.DataFrame(results)

                # Find the best and worst models
                max_value = results_df[metric_name].max()
                min_value = results_df[metric_name].min()

                # Highlight rows in the table
                def highlight_row(row):
                    if row[metric_name] == max_value:
                        return ['background-color: blue; color: white'] * len(row)
                    elif row[metric_name] == min_value:
                        return ['background-color: red; color: white'] * len(row)
                    return [''] * len(row)

                styled_df = results_df.style.apply(highlight_row, axis=1)
                st.table(styled_df)

                # Plot results with color-coded bars
                st.subheader(f"Model {metric_name} Comparison")
                fig, ax = plt.subplots()

                # Assign colors based on performance
                colors = results_df[metric_name].apply(
                    lambda x: 'blue' if x == max_value else ('red' if x == min_value else 'skyblue')
                )

                ax.barh(results_df["Model"], results_df[metric_name], color=colors)
                ax.set_xlabel(metric_name)
                ax.set_ylabel("Model")
                ax.set_title(f"{metric_name} of Each Algorithm")
           
                #box sa whole plot
                bbox = FancyBboxPatch(
                    (-0.4, 0.04),  # Bottom-left corner of the box (relative to figure)
                    1.4, 0.9,      # Width and height (relative to figure)
                    boxstyle="round,pad=0.05",  # Rounded corners with padding
                    transform=fig.transFigure,  # Use figure coordinates
                    edgecolor='black', linewidth=2, facecolor='none'
                )
                fig.patches.append(bbox)    
                st.pyplot(fig)

                # Save best model
                best_model_name = results_df.loc[results_df[metric_name].idxmax(), "Model"]
                st.session_state.trained_model = models[best_model_name]
                st.success(f"Best model: {best_model_name}")

                
                # Step 3: Model Hyperparameter Tuning
                st.write("<br><br>", unsafe_allow_html=True)
                st.subheader("Step 3: Model Hyperparameter Tuning")
                st.write(f"Best model: {best_model_name}")
                if "trained_model" not in st.session_state or st.session_state.trained_model is None:
                    st.warning("Please train models in Step 2 first.")
                else:
                    X_train = st.session_state.X_train
                    y_train = st.session_state.y_train

                    # Subset the training data for faster tuning (e.g., 20% of the original training data)
                    from sklearn.model_selection import train_test_split
                    X_sample, _, y_sample, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

                    

                    # Define hyperparameter grids for supported models
                    param_grids = {
                        "Support Vector Machines (SVM)": {
                            "kernel": ["linear", "rbf", "poly", "sigmoid"],
                            "C": [0.1, 1, 10],
                            "gamma": ["scale", "auto"]
                        },
                        "Random Forest": {
                            "n_estimators": [10, 50, 100],
                            "max_depth": [None, 10, 20, 30],
                            "min_samples_split": [2, 5, 10],
                        },
                        "Gradient Boosting Machines (AdaBoost)": {
                            "n_estimators": [10, 50, 100],
                            "learning_rate": [0.1, 0.2, 0.3],
                            
                        },
                        "K-Nearest Neighbors (K-NN)": {
                            "n_neighbors": [3, 5, 7, 9],
                            "weights": ["uniform", "distance"],
                        },
                        "Decision Tree (CART)": {
                            "criterion": ["gini", "entropy"],
                            "max_depth": [None, 10, 20, 30],
                            "min_samples_split": [2, 5, 10],
                        },
                        "Naive Bayes": {
                            "priors": [None, [0.25, 0.25, 0.25]],  # Example of setting priors
                            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
                        },
                        "Perceptron": {
                            "max_iter": [1, 100, 7],
                            "random_state": [1, 100, 7],
                            "eta0": [0.001, 10.0, 1.0],
                            "tol": [0.0001, 1.0, 1e-3],  # Fixed parameter name (removed extra space)
                        },
                        "Multi-Layer Perceptron (MLP)" : {
                            "random_state": [1, 100, 7],
                            "hidden_layer_sizes": [25, 50, 75, 100],
                            "activation": ["identity", "logistic", "tanh", "relu"],
                            "max_iter": [1, 100, 7],
                        },
                        "Logistic Regression" : {
                            "random_state": [1, 100, 7],
                            "C": [0.01, 1.0, 10.0],
                            "solver": ["lbfgs", "liblinear", "sag", "saga", "newton-cg"],
                            "max_iter": [1, 100, 7],
                        }
                    }
                    if best_model_name in param_grids:
                        param_grid = param_grids[best_model_name]
                        param_combinations = list(islice(ParameterGrid(param_grid), 4))
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        best_score = -float("inf")
                        best_params = None
                        tuning_results = []

                        for idx, params in enumerate(param_combinations):
                           # Update progress
                            progress = (idx + 1) / len(param_combinations)
                            progress_bar.progress(progress)
                            progress_text.text(f"Evaluating {idx + 1}/{len(param_combinations)} combinations...")

                            # Set parameters and train
                            model = models[best_model_name].set_params(**params)
                            model.fit(X_sample, y_sample)
                            score = model.score(X_sample, y_sample)

                            tuning_results.append({"Hyperparameters": params, "Accuracy": score})

                            # Update best parameters
                            if score > best_score:
                                best_score = score
                                best_params = params

                        
                        results_df = pd.DataFrame(tuning_results)
                        results_df["Accuracy"] = results_df["Accuracy"].apply(lambda x: f"{x * 100:.2f}%")
                        progress_text.text("Hyperparameter tuning complete!")
                        progress_bar.progress(1.0)
                        
                        st.subheader("Hyperparameter Tuning Results")
                        st.write(results_df)
                        st.success(f"Best Hyperparameters: {best_params}")
                        st.write(f"Best Accuracy: {best_score * 100:.2f}%")

                        # Save the tuned model
                        st.session_state.trained_model = models[best_model_name].set_params(**best_params)
                        st.session_state.trained_model.fit(X_train, y_train)
                        st.session_state.best_model_name = best_model_name
                        st.success(f"The best-tuned {best_model_name} model has been saved for use in Step 4 on Model Usage.")

                    else:
                        st.warning(f"No hyperparameter tuning defined for {best_model_name}.")

# Step 4: Model Usage
elif step == "Model Usage":
    st.subheader("Step 4: Model Usage")

    if st.session_state.trained_model is None:
        st.warning("Please train a model in Step 2 first.")
    else:
        model = st.session_state.trained_model
        X_test = st.session_state.get("X_test", None)
        y_test = st.session_state.get("y_test", None)

        if X_test is None or y_test is None:
            st.error("Test data not found. Please complete Steps 1 and 2 first.")
        else:
            st.markdown(
            "<h3 style='font-size: 23px; margin-top:25px;text-align:center;'>Predictions and Evaluation</h3>", 
            unsafe_allow_html=True
                )
           
            try:
                # Make predictions
                y_pred = model.predict(X_test)

                col1, col2 = st.columns(2)

                # Data Info in the first column
                with col1:
                    # Display Predictions
                    st.write("Predictions on Test Data")
                    prediction_df = pd.DataFrame({
                        "Actual": y_test.values if hasattr(y_test, "values") else y_test,
                        "Predicted": y_pred
                    })
                    st.write(prediction_df.head())

                # Missing Values in the second column
                with col2:
                    # Classification report metrics
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.write("Classification Report")
                    report_df = pd.DataFrame(report).transpose()
                    st.write(report_df)

                # Interpretation of the classification report
                st.markdown(
                "<h3 style='font-size: 23px; margin-top:25px;'>Interpretation of the Classification Report</h3>", 
                unsafe_allow_html=True
                    )
               
                st.write("""
                - **Precision**: This indicates the proportion of true positive predictions made by the model out of all positive predictions. A higher precision means fewer false positives.
                - **Recall (Sensitivity)**: This metric represents the proportion of actual positive cases that were correctly identified by the model. A higher recall means the model is better at capturing positive cases and has fewer false negatives.
                - **F1 Score**: The F1 score is the harmonic mean of precision and recall. It provides a balance between the two metrics, and a higher F1 score indicates a better model performance.
                - **Support**: This shows the number of actual occurrences of the class in the specified dataset. It provides context to the precision and recall metrics, indicating how many samples of each class were present in the test set.
                """)
                st.write("<br>", unsafe_allow_html=True)

                # ROC Curve and AUC (if applicable)
                if hasattr(model, "predict_proba") and len(set(y_test)) == 2:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = roc_auc_score(y_test, y_prob)

                    st.subheader("ROC Curve")
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")                    
                    ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title("Receiver Operating Characteristic")
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                    st.subheader("Interpretation of the Result")
                    st.write(
                        """
                        - **ROC Curve**: The ROC curve visualizes the model's performance across different thresholds. It shows the trade-off between the True Positive Rate (Sensitivity) and the False Positive Rate (1 - Specificity).
                        - **AUC (Area Under the Curve)**: AUC represents the overall ability of the model to discriminate between positive and negative classes. A higher AUC indicates better model performance:
                            - **AUC = 1**: Perfect classification.
                            - **0.7 < AUC < 0.8**: Fair classification.
                            - **0.8 < AUC < 0.9**: Good classification.
                            - **AUC = 0.5**: Model performs no better than random guessing.
                        """
                    )
               
                else:
                    st.warning("ROC-AUC is only applicable for binary classification.")

                st.write("<br><br>", unsafe_allow_html=True)

                # Cross-validation
                num_folds = st.slider("Select number of folds:", 2, min(20, len(X_test)), 10)
                if len(X_test) < num_folds:
                    st.write("The dataset is too small for the number of folds selected.")
                else:
                    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=None)

                    # Calculate classification accuracy
                    results = cross_val_score(model, X_test, y_test, cv=kfold, scoring='accuracy')

                    # Check for NaN results
                    if any(pd.isna(results)):
                        st.write("Error in cross-validation results. Please check your dataset.")
                    else:
                        st.subheader("Cross-Validation Results")
                        st.write(f"Mean Accuracy: {results.mean() * 100:.3f}%")
                        st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

                        plt.figure(figsize=(10, 5))
                        plt.boxplot(results)
                        plt.title(f'{num_folds}-Fold Cross-Validation Accuracy')
                        plt.ylabel('Accuracy')
                        plt.xticks([1], [f'{num_folds}-Fold'])
                        st.pyplot(plt)

                        # Interpretation of cross-validation results
                        st.subheader("Interpretation of the Result")
                        st.write("""
                        - K-Fold Cross-Validation results show the model's performance.
                        - Mean accuracy represents the average percentage of correctly predicted instances. 
                        - Standard deviation indicates variability across different folds.
                        - A high mean accuracy with a low standard deviation suggests a reliable model.
                        - A low mean accuracy may suggest issues like overfitting or inadequate feature engineering.
                        """)

                        buffer = io.BytesIO()
                        joblib.dump(model, buffer)
                        buffer.seek(0)
                        st.download_button(
                            label="Download Trained Model",
                            data=buffer,
                            file_name="trained_model.pkl",
                            mime="application/octet-stream",
                        )

                        st.success("You can now use the trained model for predictions.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

