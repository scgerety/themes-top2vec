import os
import pandas as pd
from top2vec import Top2Vec

filepath = os.path.abspath(os.path.dirname("theme_finder.ipynb"))
data_file = "PSI2025Evals.csv"
comment_file = os.path.join(filepath, data_file)

df = pd.read_csv(comment_file, header=1, skiprows=[2])
df = df.dropna(subset="Please provide any additional comments you would like to add here.")
comments = df[["Please provide any additional comments you would like to add here.","session"]]
comments.columns = ["comments","session"]

docs = comments.comments.tolist()
new_model = Top2Vec(docs, embedding_model="universal-sentence-encoder")

print(new_model)
