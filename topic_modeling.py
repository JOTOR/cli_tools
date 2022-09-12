import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
SEED= 5840

parser = argparse.ArgumentParser()
parser.add_argument("--excel_file", type=str, help="Name of file with text and categories")
parser.add_argument("--text", type=str, help="Name of column that contains the text")
parser.add_argument("--topics", type=int, help="Business Decision, Number of topics to extract from the text field")

args = parser.parse_args()

if __name__ == "__main__":
    df = pd.read_excel(args.excel_file)
    sdf = df.copy()
    df.drop_duplicates(inplace=True, subset=[args.text])
    #df w/o duplicates and sdf w/ duplicates
    pl = make_pipeline(TfidfVectorizer(max_features=1500, max_df=0.9, min_df=5, lowercase=True),
                       NMF(n_components=args.topics, random_state=SEED, max_iter=500, solver="mu"))
    
    pl.fit(df[args.text])

    W = pl.transform(sdf[args.text])
    H = pl.steps[1][1].components_
    sdf["topic"] = np.argmax(W, axis=1)

    print(pd.Series(sdf["topic"]).value_counts())
    print(pd.Series(sdf["topic"]).value_counts(normalize=True))

    words = np.array(pl.steps[0][1].get_feature_names_out())
    print("Most common words by topic:")
    for i, topic in enumerate(H):
        print("Topic {}: {}".format(i, ",".join([str(x) for x in words[topic.argsort()[-10:]]])))
    
    print("Exporting results to file -- Topic_Modeling_Output.xlsx")
    sdf.to_excel("Topic_Modeling_Output.xlsx", index=False)

    