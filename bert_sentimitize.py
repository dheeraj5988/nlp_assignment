import pandas as pd
from transformers import pipeline

df = pd.read_csv("healthygamer_gg_testdata.csv")

classifier = pipeline("sentiment-analysis")
df['sentiment'] = df['message'].apply(lambda x: classifier(x[:512])[0]['label'].lower())

df.to_csv("labeled_dataset.csv", index=False)
print("Sentiment labeling complete. Saved as labeled_dataset.csv.")