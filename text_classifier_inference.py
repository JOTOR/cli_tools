import numpy as np
import pandas as pd
import argparse
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--model_file", type=str, help="File name (with .pkl extension ) of the trained model")
parser.add_argument("--excel_file", type=str, help="File name with text and categories")
parser.add_argument("--text", type=str, help="Name of column that contains the text")

args = parser.parse_args()

if __name__ == "__main__":
    print("Loading Trained Model...")
    model = joblib.load(args.model_file)
    print("Reading Data...")
    df = pd.read_excel(args.excel_file)
    print("Predicting Categories....")
    df["Predicted Category"] = model.predict(df[args.text])
    print("Exporting results to file --Text_Classification_Output.xlsx")
    df.to_excel("Text_Classification_Output.xlsx", index=False)


