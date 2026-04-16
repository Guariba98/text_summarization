import os
import nltk
from src.data_processing import load_and_split_data
from src.visualization import plot_distributions, plot_wordcloud

# Descargar las utilidades de NLTK que necesita rouge-score más adelante
nltk.download('punkt', quiet=True)

def main():
    print("--- 1. CARGA Y PREPARACIÓN DEL DATASET ---")
    df, train_df, val_df, test_df = load_and_split_data()
    print(f"Total de ejemplos: {len(df)}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Guardamos los datasets limpios para la Fase 2
    print("\nGuardando splits en formato CSV en la carpeta 'data/'...")
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    
    print("\n--- 2. ANÁLISIS EXPLORATORIO (EDA) ---")
    print("Generando gráficas (Wordcloud e Histogramas)...")
    plot_distributions(df)
    plot_wordcloud(df)
    print("¡Gráficas guardadas con éxito en la carpeta 'data/'!")
    print("\n✅ FASE 1 COMPLETADA. Listo para entrenar el modelo.")

if __name__ == '__main__':
    main()