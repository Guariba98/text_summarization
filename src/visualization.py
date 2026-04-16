import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def plot_distributions(df, output_dir="data"):
    """Grafica la distribución de longitudes de artículos y resúmenes."""
    df_plot = df.copy()
    df_plot["article_len"] = df_plot["text"].apply(lambda x: len(str(x).split()))
    df_plot["summary_len"] = df_plot["summary"].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(12,5))
    sns.histplot(df_plot["article_len"], bins=50, color="blue", label="Artículos", kde=True)
    sns.histplot(df_plot["summary_len"], bins=50, color="orange", label="Resúmenes", kde=True)
    plt.legend()
    plt.title("Distribución de longitudes de palabras")
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "distribucion_longitudes.png"))
    plt.close()

def plot_wordcloud(df, output_dir="data"):
    """Genera un WordCloud de una muestra de los artículos."""
    text_all = " ".join(df["text"].sample(min(200, len(df))).astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(text_all)
    
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud de artículos")
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "wordcloud.png"))
    plt.close()