# enhancement_data_preprocessor.py

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    """
    A class to perform domain-specific data cleaning strategies on a pandas DataFrame.
    """

    def __init__(self, dataframe):
        """
        Initializes the DataPreprocessor with a pandas DataFrame.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame to be processed.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.df = dataframe

    def handle_missing_values(self, strategy='mean', columns=None):
        """
        Handles missing values in the DataFrame using specified strategy.

        Parameters:
        strategy (str): The strategy to handle missing values. Options are 'mean', 'median', 'mode', 'drop'.
        columns (list): List of columns to apply the strategy. Default is None, which applies to all columns.

        Returns:
        pd.DataFrame: DataFrame with missing values handled.
        """
        try:
            if columns is None:
                columns = self.df.columns

            for column in columns:
                if strategy == 'mean':
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.df.dropna(subset=[column], inplace=True)
                else:
                    raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', 'drop'.")
            logging.info(f"Missing values handled using {strategy} strategy.")
        except Exception as e:
            logging.error(f"Error handling missing values: {e}")
            raise

    def encode_categorical_data(self, columns=None):
        """
        Encodes categorical columns using one-hot encoding.

        Parameters:
        columns (list): List of categorical columns to encode. Default is None, which encodes all object type columns.

        Returns:
        pd.DataFrame: DataFrame with categorical columns encoded.
        """
        try:
            if columns is None:
                columns = self.df.select_dtypes(include=['object']).columns

            self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)
            logging.info("Categorical data encoded using one-hot encoding.")
        except Exception as e:
            logging.error(f"Error encoding categorical data: {e}")
            raise

    def remove_outliers(self, columns=None, threshold=1.5):
        """
        Removes outliers from the DataFrame using the IQR method.

        Parameters:
        columns (list): List of columns to remove outliers from. Default is None, which applies to all numeric columns.
        threshold (float): The IQR threshold to identify outliers.

        Returns:
        pd.DataFrame: DataFrame with outliers removed.
        """
        try:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns

            for column in columns:
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (threshold * IQR)
                upper_bound = Q3 + (threshold * IQR)
                self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            logging.info("Outliers removed using IQR method.")
        except Exception as e:
            logging.error(f"Error removing outliers: {e}")
            raise

    def normalize_data(self, columns=None):
        """
        Normalizes the data to a 0-1 range.

        Parameters:
        columns (list): List of columns to normalize. Default is None, which applies to all numeric columns.

        Returns:
        pd.DataFrame: DataFrame with normalized data.
        """
        try:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns

            for column in columns:
                min_val = self.df[column].min()
                max_val = self.df[column].max()
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
            logging.info("Data normalized to 0-1 range.")
        except Exception as e:
            logging.error(f"Error normalizing data: {e}")
            raise

    def get_cleaned_data(self):
        """
        Returns the cleaned DataFrame.

        Returns:
        pd.DataFrame: The cleaned DataFrame.
        """
        return self.df

# Example usage:
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': ['a', 'b', 'b', np.nan, 'a'],
        'C': [100, 200, 300, 400, 500],
        'D': [1, 2, 3, 4, 1000]
    }
    df = pd.DataFrame(data)

    # Initialize the DataPreprocessor
    preprocessor = DataPreprocessor(df)

    # Handle missing values
    preprocessor.handle_missing_values(strategy='mean')

    # Encode categorical data
    preprocessor.encode_categorical_data()

    # Remove outliers
    preprocessor.remove_outliers()

    # Normalize data
    preprocessor.normalize_data()

    # Get cleaned data
    cleaned_df = preprocessor.get_cleaned_data()
    print(cleaned_df)
```
