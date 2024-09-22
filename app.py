import streamlit as st
import pandas as pd
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier


# Create the Streamlit app
def main():
    st.header("Credit Card Fraud Detection")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the data
        data = pd.read_csv(uploaded_file)

        # Separate features and target variable
        X = data.drop('Class', axis=1)
        y = data['Class']

        # Define the pipeline steps
        steps = [('scaler', RobustScaler()),
                ('smote', SMOTE(random_state=1))]
        pipeline = Pipeline(steps=steps)
        
        # Fit and transform the data
        X_resampled = pipeline.fit_transform(X, y)

        # Split the resampled data into train and test sets using train_test_split
        x_train, x_test, y_train, y_test = train_test_split(
            X_resampled, y, test_size=0.2, random_state=1
        )
        
        # Train the GBC model
        model = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
        model.fit(x_train, y_train)

        # Make predictions
        predictions = model.predict(x_test)

        # Display the predictions
        st.subheader("Predictions")
        st.write(predictions)

if __name__ == '__main__':
    main()