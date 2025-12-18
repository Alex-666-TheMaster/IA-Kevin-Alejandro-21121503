import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ruta absoluta para guardar las imágenes (solicitada)
OUTPUT_DIR = r"C:\Users\Sears\Documents\Trabajos De IA\proyecto3\graficos"

def generar_visualizaciones():
    print("--- GENERANDO VISUALIZACIONES DEL PROYECTO ---")
    
    # 1. Cargar datos
    if not os.path.exists('dataset_completo_corregido.csv'):
        print(" Error: No encuentro el archivo 'dataset_completo_corregido.csv'")
        return

    df = pd.read_csv('dataset_completo_corregido.csv')
    
    # 2. Filtrar (Quitamos a Frankenstein para ver solo Gen Z)
    df_genz = df[df['Categoria'] == 'Generacion Z'].copy()
    print(f" Datos cargados: {len(df_genz)} registros de Generación Z.")

    # Crear carpeta para guardar las imágenes (ruta absoluta)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Configurar estilo visual y paleta
    sns.set_theme(style="whitegrid")

    # --- GRÁFICO 1: DISTRIBUCIÓN DE SENTIMIENTOS ---
    # Esto responde a: "¿Predomina la ansiedad o la esperanza?"
    # Histograma: color y tamaño modificados
    plt.figure(figsize=(12, 7))
    sns.histplot(data=df_genz, x='TonoSentimiento', bins=15, kde=True, color='#2a9d8f', edgecolor='white')
    plt.title('Distribución Emocional de la Gen Z (1=Negativo, 10=Positivo)')
    plt.xlabel('Nivel de Sentimiento')
    plt.ylabel('Cantidad de Opiniones')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_distribucion_emocional.png'), dpi=150)
    plt.clf() # Limpiar lienzo
    print(" Gráfico 1 guardado: Distribución Emocional")

    # --- GRÁFICO 2: FUENTES DE DISCURSO ---
    # Esto responde a: "¿Dónde se da la discusión? (Habermas/Espacio Público)"
    # Gráfico de barras: nueva paleta y tamaño
    plt.figure(figsize=(11, 6))
    conteo_medios = df_genz['Medio'].value_counts()
    # Usar matplotlib directamente con paleta para evitar la advertencia de seaborn
    colores = sns.color_palette("mako", n_colors=len(conteo_medios))
    plt.bar(conteo_medios.index, conteo_medios.values, color=colores, edgecolor='white')
    plt.title('Plataformas Digitales donde opina la Gen Z')
    plt.ylabel('Número de Interacciones')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_plataformas_digitales.png'), dpi=150)
    plt.clf()
    print(" Gráfico 2 guardado: Plataformas Digitales")

    # --- GRÁFICO 3: EVOLUCIÓN TEMPORAL (Crisis de sentido) ---
    # Convertir fecha
    df_genz['Fecha'] = pd.to_datetime(df_genz['Fecha'], errors='coerce')
    df_por_fecha = df_genz.groupby(df_genz['Fecha'].dt.to_period('M'))['TonoSentimiento'].mean()
    
    # Línea temporal: color y medidas ajustadas
    plt.figure(figsize=(14, 6))
    df_por_fecha.plot(kind='line', marker='o', color='#e76f51', linewidth=2.5)
    plt.title('Evolución del Estado de Ánimo Promedio en el Tiempo')
    plt.xlabel('Fecha')
    plt.ylabel('Sentimiento Promedio')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_evolucion_animo.png'), dpi=150)
    
    print(" Gráfico 3 guardado: Evolución Temporal")
    print("\n ¡Listo! Revisa la carpeta 'graficos' para ver tus imágenes.")

if __name__ == "__main__":
    generar_visualizaciones()