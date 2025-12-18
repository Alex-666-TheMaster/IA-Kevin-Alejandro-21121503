import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# Ruta absoluta para guardar la imagen de la nube
OUTPUT_DIR = r"C:\Users\Sears\Documents\Trabajos De IA\proyecto3\graficos"

def crear_nube():
    print("--- GENERANDO VISUALIZACIÓN SEMÁNTICA (NUBE DE PALABRAS) ---")
    
    # Cargar datos limpios
    if not os.path.exists('dataset_completo_corregido.csv'):
        print(" Faltan los datos.")
        return

    df = pd.read_csv('dataset_completo_corregido.csv')
    df = df[df['Categoria'] == 'Generacion Z']

    # Unir todos los comentarios en un solo texto gigante
    texto_completo = " ".join(comentario for comentario in df['ComentarioReaccion'])

    # Configurar la nube de palabras (quitando palabras comunes irrelevantes)
    stopwords_es = ["de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un", "por", "con", "no", "una", "su", "para", "es", "al", "lo", "como", "mas", "pero", "sus", "le", "ya", "o", "porque", "muy", "sin", "sobre", "tambien", "me", "hasta", "donde", "quien", "desde", "nos", "durante", "uno", "ni", "contra", "ese", "eso", "mí", "mis", "tengo", "esta", "estamos"]

    # Paleta personalizada y función de color para variar el diseño
    palette = ['#e9c46a', '#f4a261', '#e76f51', '#2a9d8f', '#264653']

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return palette[abs(hash(word)) % len(palette)]

    # Generar nube con fondo oscuro, más compacta y fuentes distintas
    wordcloud = WordCloud(
        width=1400,
        height=800,
        background_color='#0f172a',
        stopwords=stopwords_es,
        max_words=200,
        min_font_size=8,
        prefer_horizontal=0.9,
        random_state=42
    ).generate(texto_completo)

    # Aplicar colores personalizados
    wordcloud_recolored = wordcloud.recolor(color_func=color_func)

    # Mostrar y guardar
    plt.figure(figsize=(18, 9), facecolor='#0f172a')
    plt.imshow(wordcloud_recolored)
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, '4_nube_semantica.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    print(f" Nube de palabras guardada en '{out_path}'")

if __name__ == "__main__":
    crear_nube()