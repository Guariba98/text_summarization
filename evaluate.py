import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Cargando tu modelo entrenado... (esto puede tardar unos segundos)")
MODEL_PATH = "./models/t5_bbc_summary"

# Cargamos TU modelo, no el de internet
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

def resumir_texto(texto):
    if not texto or len(texto.strip()) == 0:
        return "Por favor, introduce un texto válido."
    
    # T5 siempre necesita que le digamos su orden al principio
    texto_preparado = "summarize: " + texto
    
    # Convertimos el texto a tokens (números)
    inputs = tokenizer(texto_preparado, return_tensors="pt", max_length=512, truncation=True)
    
    # Le pedimos al modelo que genere la respuesta
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,      # Longitud máxima del resumen
        min_length=30,       # Longitud mínima
        length_penalty=2.0,  # Penaliza si se queda muy corto
        num_beams=4,         # Explora 4 caminos posibles para que la frase suene mejor
        early_stopping=True
    )
    
    # Traducimos los números generados de vuelta a texto humano
    resumen = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return resumen

# Diseñamos la web
interfaz = gr.Interface(
    fn=resumir_texto,
    inputs=gr.Textbox(lines=10, placeholder="Pega aquí la noticia en inglés que quieras resumir...", label="Texto Original (Máx 512 palabras)"),
    outputs=gr.Textbox(lines=4, label="Resumen Generado por tu IA"),
    title="📝 Resumidor Automático de Noticias (T5 Fine-Tuned)",
    description="Modelo entrenado localmente con el dataset de la BBC. Introduce un texto en inglés y obtén un resumen al instante.",
    theme="default"
)

if __name__ == "__main__":
    print("¡Interfaz lista! Abriendo en tu navegador local...")
    # Lanza el servidor web local
    interfaz.launch(inbrowser=True)