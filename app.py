import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# Configuración de la página
st.set_page_config(page_title='Reconocimiento de Imágenes v1.0', layout="wide", page_icon="📷")

# CSS personalizado - Tema Moderno Mejorado
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Tema principal */
        body {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: 'Poppins', sans-serif;
        }
        .stApp {
            background-color: #1e293b;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.3);
            margin: 1rem auto;
            max-width: 1200px;
        }
        
        /* Encabezados */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: white !important;
        }
        
        /* Barra lateral */
        .stSidebar {
            background: linear-gradient(180deg, #1e293b, #0f172a);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        
        /* Texto normal */
        .stText {
            color: #f8fafc !important;
        }
        
        /* Cámara */
        .stCamera {
            border-radius: 16px;
            overflow: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# Título en blanco con estilo
st.markdown("<h1 style='color: white; text-align: center;'>📷 Reconocimiento de Imágenes</h1>", unsafe_allow_html=True)

# Versión de Python
st.write("Versión de Python:", platform.python_version())

# Cargar modelo
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Imagen de ejemplo
image = Image.open('image.jpg')
st.image(image, width=350)

# Barra lateral
with st.sidebar:
    st.subheader("Clasificación de Imágenes")
    st.write("Usando un modelo entrenado en Teachable Machine puedes identificar objetos en tiempo real.")
    st.write("Toma una foto con la cámara y la app te dirá qué detecta.")

# Entrada de cámara
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)
    
    # Procesamiento de imagen
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)
    
    # Normalización
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    
    # Predicción
    prediction = model.predict(data)
    st.markdown("---")
    
    # Resultados
    if prediction[0][0] > 0.5:
        st.markdown(f"<h2 style='color: #7dd3fc;'>Izquierda, con Probabilidad: {prediction[0][0]:.2f}</h2>", unsafe_allow_html=True)
    if prediction[0][1] > 0.5:
        st.markdown(f"<h2 style='color: #7dd3fc;'>Arriba, con Probabilidad: {prediction[0][1]:.2f}</h2>", unsafe_allow_html=True)

# Pie de página
st.markdown("---")
st.caption("Aplicación desarrollada con Streamlit y Keras | © 2023 Reconocimiento de Imágenes")
