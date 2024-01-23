import cv2
import streamlit as st
from ultralytics import YOLO
import os
import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape

from io import BytesIO

CARPETA_CARGA = 'uploads'

# Diccionario para contabilizar detecciones por clase
cont_detecciones_clase = {}

st.set_page_config(page_title="Cargar Imagen o Video")

def generar_frames(ruta_imagen, contenedor_detecciones):
    modelo = YOLO("tomate2.pt")

    imagen = cv2.imread(ruta_imagen)
    imagen = cv2.resize(imagen, (640, 640))
    resultados = modelo.predict(imagen, conf=0.2)
    resultado = resultados[0]

    imagen_cv = cv2.cvtColor(resultado.plot()[:, :, ::-1], cv2.COLOR_RGB2BGR)

    # Obtener información de conteo de clases desde la cadena de texto
    cont_detecciones_str = resultado.verbose()
    cont_detecciones = extraer_cont_detecciones(cont_detecciones_str)

    # Calcular la suma total de las detecciones
    cont_total = sum(cont_detecciones.values())

    st.image(imagen_cv, channels="BGR", use_column_width=True)
    
    # Actualizar la información de contabilización en tiempo real por clase
    contenedor_detecciones.text("Detecciones por Clase:")

    # Calcular y mostrar los porcentajes relativos en una tabla
    tabla_porcentajes = {
        nombre_clase: f"{(cont / cont_total) * 100:.2f}%" for nombre_clase, cont in cont_detecciones.items()
    }

    contenedor_detecciones.write("Porcentajes Relativos:")
    contenedor_detecciones.write(tabla_porcentajes)

    generar_informe_pdf(ruta_imagen, cont_detecciones, tabla_porcentajes)


def generar_video_frames(ruta_video, contenedor_detecciones):
    modelo = YOLO("tomate2.pt")

    cap = cv2.VideoCapture(ruta_video)

    st.warning("Mostrando frames del video:")

    contenedor_frames = st.empty()

    cont_total = {}  # Inicializamos un diccionario para llevar un seguimiento acumulativo de los conteos

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Obtener el total de frames del video

    generar_informe = False  # Variable de estado para controlar la generación del informe

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))
        resultados = modelo.predict(frame, conf=0.2)
        resultado = resultados[0]

        imagen_cv = cv2.cvtColor(resultado.plot()[:, :, ::-1], cv2.COLOR_RGB2BGR)

        # Obtener información de conteo de clases desde la cadena de texto
        cont_detecciones_str = resultado.verbose()
        cont_detecciones = extraer_cont_detecciones(cont_detecciones_str)

        contenedor_frames.image(imagen_cv, channels="BGR", use_column_width=True)

        # Actualizar la información de contabilización en tiempo real por clase
        contenedor_detecciones.text("Detecciones por Clase:")

        # Actualizar los conteos acumulativos
        for nombre_clase, cont in cont_detecciones.items():
            cont_total[nombre_clase] = cont_total.get(nombre_clase, 0) + cont

        # Calcular la suma total de las detecciones
        cont_total_global = sum(cont_total.values())

        # Calcular y mostrar los porcentajes relativos en una tabla
        tabla_porcentajes_global = {
            nombre_clase: f"{(cont / cont_total_global) * 100:.2f}%" for nombre_clase, cont in cont_total.items()
        }

        contenedor_detecciones.write("Porcentajes Relativos:")
        contenedor_detecciones.write(tabla_porcentajes_global)

        # Agregar botón para generar informe PDF solo en el último frame
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == total_frames - 1 and not generar_informe:
            boton_key = f"boton_informe_pdf_{cap.get(cv2.CAP_PROP_POS_FRAMES)}"
            if st.button("Generar Informe PDF", key=boton_key):
                generar_informe = True  # Establecer la variable de estado
                break  # Salir del bucle

    # Después del bucle, si la variable de estado indica que debemos generar el informe, lo generamos
    if generar_informe:
        generar_informe_pdf(ruta_video, cont_total, tabla_porcentajes_global)
        st.stop()  # Interrumpir la ejecución de la aplicación


def extraer_cont_detecciones(cont_detecciones_str):
    cont_detecciones = {}

    # Buscar todas las ocurrencias de patrones "número espacio palabra"
    coincidencias = re.findall(r'(\d+)\s+(\w+)', cont_detecciones_str)
    
    for cont, nombre_clase in coincidencias:
        cont_detecciones[nombre_clase] = int(cont)

    return cont_detecciones


def generar_informe_pdf(ruta_archivo, cont_detecciones, tabla_porcentajes):
    # Crear un objeto BytesIO para almacenar el PDF
    buffer_pdf = BytesIO()

    # Crear el objeto PDF con tamaño de página A4
    pdf = canvas.Canvas(buffer_pdf, pagesize=letter)

    # Obtener el tamaño de la página
    ancho_pagina, alto_pagina = pdf._pagesize

    # Agregar título
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(ancho_pagina / 2, alto_pagina - 30, "Informe de Detección")

    # Agregar imagen o frame al PDF
    ancho_imagen = 320  # Ancho deseado de la imagen en el PDF
    alto_imagen = 320  # Altura deseada de la imagen en el PDF
    x_imagen = (ancho_pagina - ancho_imagen) / 2  # Centrar horizontalmente
    y_imagen = alto_pagina - 30 - 50 - alto_imagen  # Dejar espacio para el título y la tabla

    pdf.drawImage(ruta_archivo, x_imagen, y_imagen, width=ancho_imagen, height=alto_imagen)

    # Agregar tabla con detecciones porcentuales
    pdf.setFont("Helvetica", 12)
    pdf.drawString(x_imagen, y_imagen - 30, "Detecciones por Clase:")  # Ajustar la posición vertical

    altura_fila = 20
    x = x_imagen
    y = y_imagen - 50  # Ajustar la posición vertical

    for nombre_clase, cont in cont_detecciones.items():
        pdf.drawString(x, y, f"{nombre_clase}: {cont} detecciones - {tabla_porcentajes[nombre_clase]}")
        y -= altura_fila

    # Guardar el PDF
    pdf.save()

    # Descargar el PDF
    st.download_button(
        label="Descargar Informe PDF",
        data=buffer_pdf.getvalue(),
        file_name="informe_deteccion.pdf",
        key="pdf_report",
    )


if __name__ == "__main__":
    st.title("Cargar Imagen o Video")

    opcion = st.sidebar.radio("Seleccionar Procesamiento", ["Imagen", "Video"])

    archivo_cargado = st.file_uploader(f"Elegir un archivo {opcion.lower()}", type=["jpg", "jpeg", "png", "mp4"])

    if archivo_cargado is not None:
        ruta_archivo = os.path.join(CARPETA_CARGA, archivo_cargado.name)
        with open(ruta_archivo, "wb") as f:
            f.write(archivo_cargado.getvalue())

        # Crear un contenedor para la información de contabilización por clase
        contenedor_detecciones = st.sidebar.empty()

        if opcion == "Imagen":
            st.image(ruta_archivo, channels="BGR", use_column_width=True)
            st.warning("Mostrando imagen procesada por YOLO:")
            generar_frames(ruta_archivo, contenedor_detecciones)
        elif opcion == "Video":
            generar_video_frames(ruta_archivo, contenedor_detecciones)
