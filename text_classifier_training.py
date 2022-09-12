import numpy as np
import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from time import strftime
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--excel_file", type=str, help="Name of file with text and categories")
parser.add_argument("--text", type=str, help="Name of column that contains the text")
parser.add_argument("--target", type=str, help="Name of column that contains the categories/target")

args = parser.parse_args()

if __name__ == "__main__":
    df = pd.read_excel(args.excel_file)
    X = df[args.text]
    y = df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=6810, test_size=0.25)

    pl = make_pipeline(TfidfVectorizer(max_features=5000, max_df=0.95, min_df=3, lowercase=True, ngram_range=(1,1), analyzer="word"),
                       LinearSVC(C=0.1, class_weight="balanced", max_iter=2500, random_state=6810))
    
    params = {"tfidfvectorizer__max_features":[1000, 2500, 5000, 7500, 10000, 20000],
              "tfidfvectorizer__ngram_range":[(1,1), (1,3), (1,5), (2,2),(3,3), (5,5)],
              "tfidfvectorizer__analyzer":["word", "char", "char_wb"]
            }

    print("Class Distribution:")
    print(pd.Series(y).value_counts())
    gs = RandomizedSearchCV(estimator=pl, param_distributions=params, cv=3, n_iter=15 , verbose=2, random_state=6810)
    print("Random Search for best model:")
    gs.fit(X_train, y_train)
    print("Best model score:")
    print(gs.best_estimator_.score(X_test, y_test))
    preds = gs.best_estimator_.predict(X_test)
    print("Performance Metrics:")
    print(classification_report(y_test, preds))
    print("---------------")
    print(confusion_matrix(y_test, preds))
    print("---------------")
    print("Exporting error analysis report"+"--"+"Error_Analysis_Report.xlsx")
    results = pd.DataFrame()
    results[args.text] = X_test
    results["actual_category"] = y_test
    results["predicted_category"] = preds
    results.to_excel("Error_Analysis_Report.xlsx", index=False)
    
    print("Training best model with entire dataset")
    gs.best_estimator_.fit(X, y)
    EXP_FILE_NAME = "Text_Classifier_"+str(datetime.now().strftime(format="%Y_%m_%d_%H_%M_%S"))+".pkl"
    print("Exporting trained model "+"--"+EXP_FILE_NAME)
    joblib.dump(gs.best_estimator_, filename=EXP_FILE_NAME)
    print("Model Training has been Completed!!!!!")