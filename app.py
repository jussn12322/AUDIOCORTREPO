"""
Aplicación de Streamlit para Limpiar Silencios y Extraer Audio
Autor: Desarrollador Senior de Python
Descripción: Herramienta web que procesa videos o audios para eliminar silencios
             y exportar el resultado como MP3.
"""

import streamlit as st
import os
import tempfile
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


# ======================== CONFIGURACIÓN DE LA PÁGINA ========================
st.set_page_config(
    page_title="AudioCort - Limpiador de Silencios",
    page_icon="🎵",
    layout="centered"
)


# ======================== TÍTULO Y DESCRIPCIÓN ========================
st.title("🎵 AudioCort - Limpiador de Silencios")
st.markdown("""
### Extrae audio y elimina silencios automáticamente
Sube un **video** (mp4, mov, mkv) o **audio** (mp3, wav, m4a) y obtén un archivo MP3
limpio sin pausas largas. Perfecto para podcasts, clases grabadas o presentaciones.
""")

st.divider()


# ======================== SIDEBAR CON CONTROLES ========================
st.sidebar.header("⚙️ Configuración de Procesamiento")
st.sidebar.markdown("Ajusta los parámetros para detectar y eliminar silencios:")

# Umbral de silencio en decibelios (dB)
# Valores más bajos (ej: -60) detectan más silencios
# Valores más altos (ej: -30) solo detectan silencios muy profundos
silence_thresh = st.sidebar.slider(
    "🔇 Umbral de Silencio (dB)",
    min_value=-80,
    max_value=-10,
    value=-40,
    step=5,
    help="Nivel de volumen considerado como silencio. Más negativo = más sensible."
)

# Duración mínima de silencio en milisegundos
# Solo se eliminarán silencios que duren al menos este tiempo
min_silence_len = st.sidebar.slider(
    "⏱️ Duración Mínima de Silencio (ms)",
    min_value=100,
    max_value=2000,
    value=500,
    step=50,
    help="Silencios más cortos que esto se mantendrán en el audio."
)

# Margen de seguridad (padding) antes/después de cada segmento de audio
# Evita cortes bruscos manteniendo un pequeño margen
keep_silence = st.sidebar.slider(
    "🛡️ Padding/Margen (ms)",
    min_value=0,
    max_value=500,
    value=150,
    step=50,
    help="Milisegundos de silencio a mantener antes/después de cada segmento."
)

st.sidebar.divider()
st.sidebar.info("💡 **Tip:** Si el audio resultante tiene cortes bruscos, aumenta el Padding/Margen.")


# ======================== CARGA DE ARCHIVO ========================
st.subheader("📁 Paso 1: Sube tu archivo")

uploaded_file = st.file_uploader(
    "Selecciona un archivo de video o audio",
    type=["mp4", "mov", "mkv", "mp3", "wav", "m4a"],
    help="Formatos soportados: MP4, MOV, MKV, MP3, WAV, M4A"
)


# ======================== PROCESAMIENTO ========================
if uploaded_file is not None:
    st.success(f"✅ Archivo cargado: **{uploaded_file.name}**")

    # Determinar el tipo de archivo
    file_extension = uploaded_file.name.split(".")[-1].lower()
    is_video = file_extension in ["mp4", "mov", "mkv"]

    if is_video:
        st.info("🎬 Video detectado. Se extraerá la pista de audio automáticamente.")
    else:
        st.info("🎵 Archivo de audio detectado. Se procesará directamente.")

    # Botón de procesamiento
    if st.button("🚀 Procesar y Limpiar Silencios", type="primary"):

        # Crear archivos temporales para el procesamiento
        temp_input = None
        temp_output = None

        try:
            with st.spinner("⏳ Procesando... Esto puede tomar unos momentos."):

                # ============ PASO 1: Guardar archivo subido temporalmente ============
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_input:
                    tmp_input.write(uploaded_file.read())
                    temp_input = tmp_input.name

                st.text("📥 Archivo guardado temporalmente...")


                # ============ PASO 2: Cargar audio ============
                # AudioSegment.from_file() es muy inteligente:
                # - Detecta automáticamente el formato del archivo (mp4, mp3, wav, etc.)
                # - Si es un VIDEO (mp4/mov/mkv), extrae la pista de audio automáticamente
                # - Si es AUDIO, lo carga directamente
                # - Requiere ffmpeg instalado en el sistema para formatos como mp4, m4a
                st.text("🎧 Cargando y extrayendo audio...")
                audio = AudioSegment.from_file(temp_input)


                # ============ PASO 3: Detectar segmentos NO silenciosos ============
                # detect_nonsilent() devuelve una lista de tuplas (inicio, fin) en ms
                # Cada tupla representa un segmento de audio que NO es silencio
                st.text("🔍 Detectando silencios...")
                nonsilent_ranges = detect_nonsilent(
                    audio,
                    min_silence_len=min_silence_len,  # Duración mínima del silencio
                    silence_thresh=silence_thresh,     # Umbral de volumen
                    seek_step=1                        # Precisión de búsqueda (1 ms)
                )

                # Verificar si se encontraron segmentos de audio
                if not nonsilent_ranges:
                    st.warning("⚠️ No se detectaron segmentos de audio. El archivo podría estar completamente en silencio.")
                    st.stop()


                # ============ PASO 4: Unir segmentos no silenciosos ============
                st.text("✂️ Eliminando silencios y uniendo segmentos...")

                # Crear un nuevo AudioSegment vacío para almacenar el resultado
                cleaned_audio = AudioSegment.empty()

                # Iterar sobre cada segmento no silencioso
                for start, end in nonsilent_ranges:
                    # Aplicar padding (margen de seguridad) antes y después
                    # max(0, ...) evita valores negativos en el inicio
                    start_with_padding = max(0, start - keep_silence)
                    end_with_padding = min(len(audio), end + keep_silence)

                    # Extraer el segmento con padding y añadirlo al resultado
                    segment = audio[start_with_padding:end_with_padding]
                    cleaned_audio += segment


                # ============ PASO 5: Exportar como MP3 ============
                # Crear archivo temporal para el MP3 de salida
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_output:
                    temp_output = tmp_output.name

                st.text("💾 Exportando como MP3...")

                # Exportar con configuración de calidad
                cleaned_audio.export(
                    temp_output,
                    format="mp3",
                    bitrate="192k",  # Calidad alta (192 kbps)
                    parameters=["-q:a", "2"]  # Calidad VBR de ffmpeg (0-9, 2 es alta)
                )


                # ============ PASO 6: Mostrar resultados ============
                st.success("✅ ¡Procesamiento completado exitosamente!")

                # Estadísticas
                original_duration = len(audio) / 1000  # Convertir a segundos
                cleaned_duration = len(cleaned_audio) / 1000
                time_saved = original_duration - cleaned_duration

                col1, col2, col3 = st.columns(3)
                col1.metric("⏱️ Duración Original", f"{original_duration:.1f}s")
                col2.metric("⏱️ Duración Final", f"{cleaned_duration:.1f}s")
                col3.metric("💨 Tiempo Ahorrado", f"{time_saved:.1f}s")

                st.divider()


                # ============ PASO 7: Reproductor de audio ============
                st.subheader("🎧 Paso 2: Escucha el resultado")

                with open(temp_output, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")

                st.divider()


                # ============ PASO 8: Descarga ============
                st.subheader("📥 Paso 3: Descarga tu archivo")

                # Generar nombre de archivo de salida
                original_name = os.path.splitext(uploaded_file.name)[0]
                output_filename = f"{original_name}_limpio.mp3"

                st.download_button(
                    label="⬇️ Descargar MP3 Limpio",
                    data=audio_bytes,
                    file_name=output_filename,
                    mime="audio/mp3",
                    type="primary"
                )

                st.success(f"💾 Listo para descargar: **{output_filename}**")


        # ============ MANEJO DE ERRORES ============
        except Exception as e:
            st.error(f"❌ Error durante el procesamiento: {str(e)}")
            st.info("""
            **Posibles causas:**
            - El archivo está corrupto o no es válido
            - Formato de archivo no soportado correctamente
            - Problema con ffmpeg (necesario para procesar videos)

            **Soluciones:**
            - Intenta con otro archivo
            - Verifica que el archivo no esté dañado
            - Asegúrate de que ffmpeg esté instalado
            """)


        # ============ LIMPIEZA DE ARCHIVOS TEMPORALES ============
        finally:
            # Eliminar archivos temporales para no saturar el disco
            if temp_input and os.path.exists(temp_input):
                try:
                    os.unlink(temp_input)
                except:
                    pass  # Si no se puede eliminar, no es crítico

            if temp_output and os.path.exists(temp_output):
                try:
                    os.unlink(temp_output)
                except:
                    pass


# ======================== PIE DE PÁGINA ========================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
        🚀 Desarrollado con Streamlit y pydub |
        💡 Requiere ffmpeg para procesamiento de videos
    </small>
</div>
""", unsafe_allow_html=True)
