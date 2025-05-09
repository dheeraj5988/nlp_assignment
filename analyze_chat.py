import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('labeled_dataset.csv')

def generate_wordcloud(sentiment_label):
    text = ' '.join(df[df['sentiment'] == sentiment_label]['message'])
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud for {sentiment_label} sentiment")
    plt.show()

generate_wordcloud("positive")
generate_wordcloud("negative")