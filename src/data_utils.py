import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(raw_data, mitigation=False):
    data = raw_data[raw_data['birth date'] < 2020].copy()
    data['age'] = data['parole board interview date'] - data['birth date']
    data['jail duration'] = data['parole board interview date'] - data['year of entry']
    others_parole_type =  ['PIE', 'SP CONSDR', 'ECPDO', 'MEDICAL','RESCISSION', 'DEPORT']
    data['parole board interview type'] = data['parole board interview type'].replace(others_parole_type, 'OTHERS').replace('SUPP MERIT', 'MERIT TIME').replace('PV REAPP', 'REAPPEAR')
    data = data.dropna(axis=0, subset=['crime 1 - class', 'parole eligibility date'])

    if mitigation:
        data = data.drop(columns={"sex", "race / ethnicity"})
    else:
        df_one_hot = pd.get_dummies(data, columns=[
            "sex", "race / ethnicity"], drop_first=True)
        
    df_one_hot = pd.get_dummies(df_one_hot, columns=[
        "crime 1 - class", "crime 2 - class",
        "crime 3 - class", "crime 4 - class",
        "parole board interview type"])
    df_one_hot.drop(columns=['release date','birth date', 'year of entry'], inplace=True)
    return df_one_hot

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.3, random_state=42)
    return X_train, X_test, y_train, y_test