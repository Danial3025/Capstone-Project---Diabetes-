import pandas as pd
import joblib 

def load_data():
    df = pd.read_csv(r"C:\Users\Acer\Desktop\Data-Science\capstone\application\data\diabetes.csv")
    return df

joblib.dump(load_data, "load_data.pkl")