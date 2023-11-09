import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(raw_data: pd.DataFrame, mitigation: bool = False) -> pd.DataFrame:
    """
    Preprocesses raw data for analysis.

    Args:
        raw_data (DataFrame): The raw data containing parole-related information.
        mitigation (bool, optional): If True, perform mitigation-specific preprocessing, such as removing 'sex' and 'race / ethnicity' columns. If False, perform general preprocessing with one-hot encoding. Defaults to False.

    Returns:
        DataFrame: The preprocessed data with one-hot encoding and appropriate columns dropped.
    """
    data = raw_data[raw_data['birth date'] < 2020].copy()
    data['age'] = data['parole board interview date'] - data['birth date']
    data['jail duration'] = data['parole board interview date'] - data['year of entry']
    others_parole_type =  ['PIE', 'SP CONSDR', 'ECPDO', 'MEDICAL','RESCISSION', 'DEPORT']
    data['parole board interview type'] = data['parole board interview type'].replace(others_parole_type, 'OTHERS').replace('SUPP MERIT', 'MERIT TIME').replace('PV REAPP', 'REAPPEAR')
    data = data.dropna(axis=0, subset=['crime 1 - class', 'parole eligibility date'])

    if mitigation:
        data = data.drop(columns={"sex", "race / ethnicity"})
        df_one_hot = data
    else:
        df_one_hot = pd.get_dummies(data, columns=[
            "sex", "race / ethnicity"], drop_first=True)
        
    df_one_hot = pd.get_dummies(df_one_hot, columns=[
        "crime 1 - class", "crime 2 - class",
        "crime 3 - class", "crime 4 - class",
        "parole board interview type"])
    df_one_hot.drop(columns=['release date','birth date', 'year of entry'], inplace=True)
    return df_one_hot

def split_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Split input data and labels into training and testing sets.

    Args:
        X (np.ndarray): Input data (features).
        y (np.ndarray): Labels (target variable).

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test