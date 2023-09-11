import openpyxl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def analyze_net_worth_data(file_path):
    data = pd.read_excel(file_path)
    
    #Create the input dataset by dropping irrelevant features
    irrelevant_inputs = ['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Healthcare Cost']
    input_data = data.drop(irrelevant_inputs, axis=1)
    
    #Create output dataset from the origianl dataset
    output_data = data['Net Worth']
    
    #Prints the chart
    sns.pairplot(data)
    plt.show()

file_path = r"C:\Users\joash\OneDrive\Documents\Techtorium\Techtorium SD 2023\Term 3 Assessment (ET)\Net_Worth_Data.xlsx"
analyze_net_worth_data(file_path)