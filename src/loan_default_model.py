# -*- coding: utf-8 -*-
"""
Loan Default Prediction Model (LDPM)
First draft - Random Forest implementation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

class LoanDefaultPredictor:
    def __init__(self, features_path='data/account_features.csv', loan_path='data/Loan.csv'):
        """
        Initialize the Loan Default Predictor with paths to necessary data files.

        Parameters:
        features_path (str): Path to the engineered features CSV
        loan_path (str): Path to the loan data CSV
        """
        self.features_path = features_path
        self.loan_path = loan_path
        self.model = None
        self.feature_columns = None

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_and_prepare_data(self):
        """
        Load features and loan data, merge them, and prepare for modeling.
        """
        try:
            # Load the data
            self.logger.info("Loading feature and loan data...")
            features_df = pd.read_csv(self.features_path)
            loan_df = pd.read_csv(self.loan_path, sep=';')

            # Create target variable (1 for default, 0 for non-default)
            loan_df['default'] = (loan_df['status'] == 'B').astype(int)

            # Merge features with loan status
            self.logger.info("Merging features with loan status...")
            model_data = pd.merge(
                features_df,
                loan_df[['account_id', 'default']],
                on='account_id',
                how='inne r'
            )

            # Select features for modeling
            self.feature_columns = [
                'transaction_frequency',
                'relationship_length',
                'credit_card_usage',
                'standing_order_coverage',
                'regional_risk_score',
                'balance_depletion_ratio'
            ]

            # Remove rows with inf or null values
            model_data = model_data.replace([np.inf, -np.inf], np.nan)
            model_data = model_data.dropna(subset=self.feature_columns + ['default'])

            X = model_data[self.feature_columns]
            y = model_data['default']

            return train_test_split(X, y, test_size=0.2, random_state=42)

        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise

    def train_model(self):
        """
        Train a Random Forest model for loan default prediction.
        """
        try:
            # Load and split the data
            self.logger.info("Preparing data for training...")
            X_train, X_test, y_train, y_test = self.load_and_prepare_data()

            # Initialize and train the model
            self.logger.info("Training Random Forest model...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_train, y_train)

            # Generate predictions
            self.logger.info("Generating predictions...")
            y_pred = self.model.predict(X_test)

            # Calculate and log performance metrics
            self.logger.info("\nModel Performance:")
            self.logger.info("\nClassification Report:")
            self.logger.info(f"\n{classification_report(y_test, y_pred)}")

            # Calculate feature importances
            importances = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            self.logger.info("\nFeature Importances:")
            self.logger.info(f"\n{importances}")

            return X_test, y_test, y_pred

        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def save_model(self, model_path='loan_default_model.joblib'):
        """
        Save the trained model to disk.
        """
        try:
            if self.model is not None:
                joblib.dump(self.model, model_path)
                self.logger.info(f"Model saved to {model_path}")
            else:
                raise ValueError("No trained model to save")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path='loan_default_model.joblib'):
        """
        Load a trained model from disk.
        """
        try:
            self.model = joblib.load(model_path)
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, features_df):
        """
        Generate predictions for new data.

        Parameters:
        features_df (pd.DataFrame): DataFrame containing the required features

        Returns:
        np.array: Array of predictions (0 for non-default, 1 for default)
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")

            # Ensure all required features are present
            missing_features = set(self.feature_columns) - set(features_df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            # Generate predictions
            predictions = self.model.predict(features_df[self.feature_columns])
            return predictions

        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = LoanDefaultPredictor()

    # Train model
    X_test, y_test, y_pred = predictor.train_model()

    # Save model
    predictor.save_model()

    # Make predictions on new data (example)
    new_predictions = predictor.predict(X_test.head())
    print("\nSample predictions:", new_predictions)