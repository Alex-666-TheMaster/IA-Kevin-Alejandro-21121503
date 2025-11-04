# ChatBot Materias ISC 
import os
import json
import unicodedata
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize
from flask import Flask, request, jsonify, render_template


# CONFIG
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "curriculum.faiss"
METADATA_PATH = "metadata.json"


# FUNCIONES BASE
def normalize_text(s: str) -> str:
    """Convierte texto a minúsculas y elimina acentos."""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = s.encode("ascii", "ignore").decode("utf-8")
    return s

class EmbeddingStore:
    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadatas = []

    def load(self):
        if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH)):
            raise FileNotFoundError("Faltan archivos de índice FAISS o metadata.json.")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadatas = json.load(f)

    def search(self, query: str, k: int = 5):
        q_vec = self.model.encode([query], convert_to_numpy=True)
        q_vec = normalize(q_vec)
        D, I = self.index.search(q_vec, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append({"score": float(score), "text": self.metadatas[idx]["text"]})
        return results


# ANALIZADOR DE MATERIAS SERIADAS
def analyze_subject(user_message: str, context: str) -> str:
    # Definición de las cadenas (seriación de ISC)
    series = [
        ["calculo diferencial", "calculo integral", "calculo vectorial", "ecuaciones diferenciales"],
        ["fundamentos de programacion", "programacion orientada a objetos", "estructura de datos", "topicos avanzados de programacion"],
        ["fundamentos de investigacion", "taller de investigacion i", "taller de investigacion ii"],
        ["matematicas discretas", "algebra lineal"],
        ["automatas i", "automatas ii"],
        ["fundamentos de telecomunicaciones", "redes de computadoras", "conmutacion y enrutamiento en redes de datos", "administracion de redes"],
        ["programacion logica y funcional", "inteligencia artificial"],
        ["principios electricos", "arquitectura de computadoras", "lenguajes de interfaz", "sistemas programables"],
        ["fundamentos de ingenieria de software", "ingenieria de software", "gestion de proyectos de software"],
        ["fundamentos de bases de datos", "taller de bases de datos", "administracion de bases de datos"],
        ["sistemas operativos", "taller de sistemas operativos"]
    ]

    user_norm = normalize_text(user_message)

    # Buscar coincidencia exacta primero
    for cadena in series:
        for i, materia in enumerate(cadena):
            mat_norm = normalize_text(materia)
            if mat_norm == user_norm:
                return build_subject_response(materia, cadena, i)

    # Si no hubo coincidencia exacta, buscar coincidencia parcial
    for cadena in series:
        for i, materia in enumerate(cadena):
            mat_norm = normalize_text(materia)
            if mat_norm in user_norm or user_norm in mat_norm:
                return build_subject_response(materia, cadena, i)

    # Si no encuentra coincidencia, usa contexto de embeddings
    resumen = context[:400].replace("\n", " ")
    return f"No encontré información específica, pero esto es lo más relacionado que encontré en la retícula:\n{resumen}..."


def build_subject_response(materia: str, cadena: list, i: int) -> str:
    """Construye la respuesta del bot para una materia dada."""
    previas = cadena[:i] if i > 0 else ["(ninguna, es base de la serie)"]
    siguientes = cadena[i+1:] if i < len(cadena)-1 else ["(ninguna, es la última de la serie)"]

    msg = f"📘 *{materia.title()}*\n\n"
    msg += f"**Pertenece a la serie:** {' → '.join(m.title() for m in cadena)}\n\n"
    msg += f"**Prerrequisitos recomendados:** {', '.join(p.title() for p in previas)}\n"
    msg += f"**Materias posteriores:** {', '.join(s.title() for s in siguientes)}\n\n"

    consejos = []

    # Cálculo y ecuaciones
    if "calculo" in materia or "ecuaciones" in materia:
        consejos = [
            "Repasa álgebra y trigonometría.",
            "Practica derivadas e integrales según el nivel.",
            "Haz ejercicios todos los días, no solo antes del examen."
        ]

    # Programación
    elif "programacion" in materia or "logica" in materia:
        consejos = [
            "Dedica tiempo diario a codificar.",
            "Resuelve problemas en plataformas como HackerRank o LeetCode.",
            "Apóyate en compañeros para revisar código y depurar errores."
        ]

    # Autómatas
    elif "automatas" in materia:
        consejos = [
            "Revisa gramáticas, expresiones regulares y teoría de conjuntos.",
            "Practica construcción de autómatas DFA y NFA.",
            "Conecta los conceptos con compiladores y análisis de lenguajes."
        ]

    # Redes y telecomunicaciones
    elif "redes" in materia or "telecomunicaciones" in materia:
        consejos = [
            "Aprende el modelo OSI y TCP/IP.",
            "Practica configuraciones básicas con Packet Tracer.",
            "Comprende direcciones IP y enrutamiento."
        ]

    # Ingeniería de software
    elif "software" in materia:
        consejos = [
            "Repasa metodologías ágiles y control de versiones.",
            "Comprende el ciclo de vida del software.",
            "Trabaja en proyectos en equipo para practicar documentación."
        ]

    # Bases de datos (ahora incluye fundamentos y taller)
    elif "bases de datos" in materia:
        consejos = [
            "Aprende a modelar con diagramas entidad-relación (E-R).",
            "Practica normalización hasta la tercera forma normal (3FN).",
            "Domina SQL con sentencias SELECT, JOIN, GROUP BY y subconsultas.",
            "Crea mini proyectos con MySQL o PostgreSQL para practicar.",
            "Si estás en el Taller, enfócate en consultas complejas y optimización de índices."
        ]

    # Álgebra y matemáticas discretas
    elif "algebra" in materia or "discretas" in materia:
        consejos = [
            "Practica operaciones con conjuntos y relaciones.",
            "Refuerza lógica proposicional y álgebra booleana.",
            "Haz ejercicios de matrices y sistemas lineales."
        ]

    # Inteligencia artificial
    elif "inteligencia artificial" in materia:
        consejos = [
            "Refuerza probabilidad, álgebra lineal y lógica.",
            "Aprende sobre búsqueda, heurísticas y redes neuronales básicas.",
            "Experimenta con librerías como Scikit-learn o TensorFlow."
        ]

    # Sistemas operativos
    elif "sistemas operativos" in materia:
        consejos = [
            "Entiende procesos, hilos, semáforos y gestión de memoria.",
            "Aprende planificación de CPU y manejo de interrupciones.",
            "Practica con Linux para observar procesos reales y comandos del sistema."
        ]

    # Principios eléctricos y arquitectura
    elif "principios electricos" in materia or "arquitectura de computadoras" in materia:
        consejos = [
            "Refuerza conceptos de corriente, voltaje y resistencia (Ley de Ohm).",
            "Aprende a leer diagramas eléctricos y simbología básica.",
            "Relaciona los conceptos con circuitos digitales y compuertas lógicas.",
            "Usa simuladores como Tinkercad o Proteus para practicar circuitos.",
            "Conecta lo aprendido con Arquitectura de Computadoras y Sistemas Programables."
        ]

    # Lenguajes de interfaz y sistemas programables
    elif "interfaz" in materia or "programables" in materia:
        consejos = [
            "Familiarízate con microcontroladores (como Arduino o PIC).",
            "Aprende el uso de entradas/salidas digitales y analógicas.",
            "Prueba códigos sencillos para activar LEDs o motores.",
            "Comprende la comunicación entre hardware y software."
        ]

    # Taller de investigación
    elif "taller de investigacion" in materia:
        consejos = [
            "Define un tema claro desde el inicio.",
            "Consulta fuentes académicas y usa normas APA o IEEE.",
            "Organiza bien la redacción: introducción, desarrollo y conclusiones.",
            "Pide retroalimentación constante a tu asesor."
        ]

    # Si hay consejos, agrégalos
    if consejos:
        msg += "**Consejos para dominarla:**\n"
        for c in consejos:
            msg += f"- {c}\n"

    return msg



    # Si no encuentra coincidencia, usa contexto de embeddings
    resumen = context[:400].replace("\n", " ")
    return f"No encontré información específica, pero esto es lo más relacionado que encontré en la retícula:\n{resumen}..."


# FLASK APP
app = Flask(__name__, template_folder="templates")
store = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global store
    if store is None:
        store = EmbeddingStore()
        store.load()

    data = request.get_json()
    user_message = data.get("message", "").strip()
    hits = store.search(user_message, k=5)
    context = "\n\n".join([h["text"] for h in hits])
    response = analyze_subject(user_message, context)
    return jsonify({"reply": response})

if __name__ == "__main__":
    print("🔥 Iniciando ChatBot ISC — versión final 2025...")
    store = EmbeddingStore()
    try:
        store.load()
        print("✅ Índice FAISS cargado correctamente.")
    except:
        print("⚠️ No se encontró el índice, genera uno con /build-index si es necesario.")
    app.run(host="0.0.0.0", port=5000, debug=True)

