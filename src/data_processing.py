import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def load_and_split_data():
    print("Descargando dataset de Hugging Face...")
    raw = load_dataset("gopalkalpande/bbc-news-summary")
    
    df = raw["train"].to_pandas()[["Articles", "Summaries"]].rename(columns={"Articles":"text", "Summaries":"summary"}) # type: ignore
    
    print("Limpiando textos...")
    df['text'] = df['text'].apply(clean_text)
    df['summary'] = df['summary'].apply(clean_text)
    

    train_df, test_df = train_test_split(df, test_size=0.10, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.10, random_state=42)
    
    return df, train_df, val_df, test_df