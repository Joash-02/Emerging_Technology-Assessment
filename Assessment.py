import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import numpy as np

class ModelFactory:
    @staticmethod
    def Accquire_model(model_name):
        if model_name == 'Linear Regression':
            return LinearRegression()
        elif model_name == 'Support Vector Machine':
            return SVR()
        elif model_name == 'Random Forest':
            return RandomForestRegressor()
        elif model_name == 'Gradient Boosting Regressor':
            return GradientBoostingRegressor()
        elif model_name == 'XGBRegressor':
            return XGBRegressor()
        elif model_name == 'Ridge Regression':
            return Ridge()
        elif model_name == 'Lasso Regression':
            return Lasso()
        else:
            raise ValueError(f"Model '{model_name}' not recognized!")

def Loads_data(file_name):
    return pd.read_excel(r"C:\Users\joash\OneDrive\Documents\Techtorium\Techtorium SD 2023\Term 3 Assessment (ET)\Net_Worth_Data.xlsx")

def Preprocess_data(data):
    if data.isnull().any().any():
        raise ValueError("The data contains missing values. Please ensure the data is cleaned before processing.")

    X = data.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Healthcare Cost'], axis=1)
    Y = data['Net Worth']
    
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    
    sc1 = MinMaxScaler()
    y_reshape = Y.values.reshape(-1, 1)
    y_scaled = sc1.fit_transform(y_reshape)
    
    return X_scaled, y_scaled, sc, sc1

def Splits_data(X_scaled, y_scaled):
    return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

def Trains_models(X_train, y_train):
    model_names = [
        'Linear Regression',
        'Support Vector Machine',
        'Random Forest',
        'Gradient Boosting Regressor',
        'XGBRegressor',
        'Ridge Regression',
        'Lasso Regression'
    ]
    
    models = {}
    for name in model_names:
        print(f"Training model: {name}")
        model = ModelFactory.Accquire_model(name)
        model.fit(X_train, y_train.ravel())
        models[name] = model
        print(f"{name} trained successfully.")
        
    return models

def Evaluating_models(models, X_test, y_test):
    rmse_values = {}
    
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse_values[name] = mean_squared_error(y_test, preds, squared=False)
        
    return rmse_values

def Ploting_model_performances(rmse_values):
    plt.figure(figsize=(10,7))
    models = list(rmse_values.keys())
    rmse = list(rmse_values.values())
    bars = plt.bar(models, rmse, color=['black'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

    plt.xlabel('NetWorth')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def retrain_model(new_data, X_train, y_train):
    X_train = np.vstack((X_train, new_data[:, :-1]))
    y_train = np.vstack((y_train, new_data[:, -1].reshape(-1, 1)))
    models = train_models(X_train, y_train)
    return models

def Save_model(models, rmse_values):
    best_model_name = min(rmse_values, key=rmse_values.get)
    best_model = models[best_model_name]
    dump(best_model, "Net_Worth.joblib")

def Predicts_data(loaded_model, sc, sc1):
    X_test1 = sc.transform(np.array([[0,42,62812.09301,11609.38091,238961.2505]]))
    pred_value = loaded_model.predict(X_test1)
    print(pred_value)

    if len(pred_value.shape) == 1:
        pred_value = pred_value.reshape(-1, 1)
    print("Predicted Outcome: ", sc1.inverse_transform(pred_value))

if __name__ == "__main__":
    try:
        data = Loads_data('Net_Worth_Data.xlsx')
        X_scaled, y_scaled, sc, sc1 = Preprocess_data(data)
        X_train, X_test, y_train, y_test = Splits_data(X_scaled, y_scaled)
        models = Trains_models(X_train, y_train)
        rmse_values = Evaluating_models(models, X_test, y_test)
        Ploting_model_performances(rmse_values)
        Save_model(models, rmse_values)
        loaded_model = load("Net_Worth.joblib")
        Predicts_data(loaded_model, sc, sc1)
    except ValueError as ve:
        print(f"Error: {ve}")
