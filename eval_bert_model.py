import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv("labeled_dataset.csv")
print(classification_report(df['sentiment'], df['sentiment']))