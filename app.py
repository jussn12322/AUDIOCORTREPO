"""
Aplicación de Streamlit para Limpiar Silencios y Extraer Audio
Autor: Desarrollador Senior de Python
Descripción: Herramienta web que procesa videos o audios para eliminar silencios
             y exportar el resultado como MP3.
Versión: 2.0 - Usa ffmpeg directamente (sin pydub)
"""

import streamlit as st
import os
import tempfile
import subprocess
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sqlite3
import base64
from datetime import datetime


# ======================== CONFIGURACIÓN DE LA PÁGINA ========================
st.set_page_config(
    page_title="AudioCort - Limpiador de Silencios",
    page_icon="🎵",
    layout="centered"
)

# ======================== INICIALIZAR SESSION STATE ========================
if 'history' not in st.session_state:
    st.session_state.history = []


# ======================== FUNCIONES DE BASE DE DATOS ========================
def init_database():
    """
    Inicializa la base de datos SQLite para almacenar el historial.
    """
    conn = sqlite3.connect('audiocort_history.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            audio_data BLOB NOT NULL,
            original_duration REAL NOT NULL,
            cleaned_duration REAL NOT NULL,
            time_saved REAL NOT NULL,
            timestamp TEXT NOT NULL,
            umbral_silencio INTEGER NOT NULL,
            duracion_minima INTEGER NOT NULL,
            padding INTEGER NOT NULL
        )
    ''')

    conn.commit()
    conn.close()


def save_to_database(filename, audio_bytes, original_duration, cleaned_duration,
                     silence_thresh, min_silence_len, keep_silence):
    """
    Guarda un audio procesado en la base de datos.
    """
    try:
        conn = sqlite3.connect('audiocort_history.db')
        cursor = conn.cursor()

        time_saved = original_duration - cleaned_duration
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute('''
            INSERT INTO history (
                filename, audio_data, original_duration, cleaned_duration,
                time_saved, timestamp, umbral_silencio, duracion_minima, padding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename, audio_bytes, original_duration, cleaned_duration,
            time_saved, timestamp, silence_thresh, min_silence_len, keep_silence
        ))

        conn.commit()

        # Mantener solo los últimos 10 registros
        cursor.execute('''
            DELETE FROM history WHERE id NOT IN (
                SELECT id FROM history ORDER BY id DESC LIMIT 10
            )
        ''')

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error al guardar en base de datos: {e}")
        return False


def load_from_database():
    """
    Carga el historial desde la base de datos.
    Retorna lista de diccionarios con la información.
    """
    try:
        conn = sqlite3.connect('audiocort_history.db')
        cursor = conn.cursor()

        cursor.execute('''
            SELECT filename, audio_data, original_duration, cleaned_duration,
                   time_saved, timestamp, umbral_silencio, duracion_minima, padding
            FROM history
            ORDER BY id DESC
            LIMIT 10
        ''')

        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append({
                'filename': row[0],
                'audio_data': row[1],
                'original_duration': row[2],
                'cleaned_duration': row[3],
                'time_saved': row[4],
                'timestamp': row[5],
                'settings': {
                    'umbral_silencio': row[6],
                    'duracion_minima': row[7],
                    'padding': row[8]
                }
            })

        return history
    except Exception as e:
        return []


def clear_database():
    """
    Limpia todo el historial de la base de datos.
    """
    try:
        conn = sqlite3.connect('audiocort_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM history')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error al limpiar base de datos: {e}")
        return False


# Inicializar base de datos al cargar la app
init_database()


# ======================== FUNCIONES AUXILIARES ========================
def get_audio_duration(file_path):
    """Obtiene la duración del audio en segundos usando ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except:
        return 0


def detect_silence_with_ffmpeg(input_file, silence_thresh_db, min_silence_duration):
    """
    Detecta silencios usando ffmpeg directamente.
    Retorna lista de tuplas (start, end) de segmentos NO silenciosos en segundos.
    """
    try:
        # Convertir umbral de dB a amplitud (ffmpeg usa 0-1)
        # -40dB ≈ 0.01, -60dB ≈ 0.001
        silence_thresh = 10 ** (silence_thresh_db / 20)

        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-af', f'silencedetect=noise={silence_thresh}:d={min_silence_duration/1000}',
            '-f', 'null',
            '-'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parsear la salida de ffmpeg para encontrar los silencios
        silence_starts = []
        silence_ends = []

        for line in result.stderr.split('\n'):
            if 'silence_start' in line:
                try:
                    start = float(line.split('silence_start: ')[1].split()[0])
                    silence_starts.append(start)
                except:
                    pass
            elif 'silence_end' in line:
                try:
                    end = float(line.split('silence_end: ')[1].split()[0])
                    silence_ends.append(end)
                except:
                    pass

        # Construir segmentos de audio (NO silencio)
        total_duration = get_audio_duration(input_file)
        nonsilent_segments = []

        # Caso especial: si no hay silencios, retornar todo el audio
        if len(silence_starts) == 0:
            return [(0, total_duration)]

        # Agregar segmento inicial si no empieza con silencio
        if silence_starts[0] > 0.1:
            nonsilent_segments.append((0, silence_starts[0]))

        # Agregar segmentos entre silencios
        for i in range(len(silence_ends)):
            if i < len(silence_starts) - 1:
                nonsilent_segments.append((silence_ends[i], silence_starts[i + 1]))

        # Agregar segmento final si no termina con silencio
        if len(silence_ends) > 0 and silence_ends[-1] < total_duration - 0.1:
            nonsilent_segments.append((silence_ends[-1], total_duration))

        return nonsilent_segments if nonsilent_segments else [(0, total_duration)]

    except Exception as e:
        st.error(f"Error al detectar silencios: {e}")
        # En caso de error, retornar todo el archivo
        duration = get_audio_duration(input_file)
        return [(0, duration)]


def merge_segments_with_ffmpeg(input_file, segments, output_file, padding_ms):
    """
    Une los segmentos de audio usando ffmpeg con padding.
    """
    try:
        # Convertir padding de ms a segundos
        padding_s = padding_ms / 1000.0

        # Crear archivo de lista temporal para ffmpeg concat
        concat_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_segments = []

        for i, (start, end) in enumerate(segments):
            # Aplicar padding
            start_with_padding = max(0, start - padding_s)
            end_with_padding = end + padding_s

            # Extraer segmento
            segment_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            segment_path = segment_file.name
            segment_file.close()
            temp_segments.append(segment_path)

            duration = end_with_padding - start_with_padding

            cmd = [
                'ffmpeg',
                '-i', input_file,
                '-ss', str(start_with_padding),
                '-t', str(duration),
                '-acodec', 'libmp3lame',
                '-b:a', '192k',
                '-y',
                segment_path
            ]

            subprocess.run(cmd, capture_output=True, check=True)
            concat_file.write(f"file '{segment_path}'\n")

        concat_file.close()

        # Concatenar todos los segmentos
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file.name,
            '-acodec', 'libmp3lame',
            '-b:a', '192k',
            '-y',
            output_file
        ]

        subprocess.run(cmd, capture_output=True, check=True)

        # Limpiar archivos temporales
        os.unlink(concat_file.name)
        for seg in temp_segments:
            try:
                os.unlink(seg)
            except:
                pass

        return True

    except Exception as e:
        st.error(f"Error al unir segmentos: {e}")
        return False


def extract_waveform_data(audio_file, duration, max_samples=3000):
    """
    Extrae datos de forma de onda del archivo de audio usando ffmpeg.
    Retorna array de amplitudes normalizadas.
    """
    try:
        # Extraer audio como datos PCM usando ffmpeg
        cmd = [
            'ffmpeg',
            '-i', audio_file,
            '-f', 's16le',  # PCM 16-bit signed little-endian
            '-ac', '1',      # Mono
            '-ar', '8000',   # Sample rate reducido para eficiencia
            '-'
        ]

        result = subprocess.run(cmd, capture_output=True, check=False)

        if result.returncode != 0:
            return None

        # Convertir bytes a array numpy
        audio_data = np.frombuffer(result.stdout, dtype=np.int16)

        # Normalizar a rango -1 a 1
        audio_data = audio_data.astype(np.float32) / 32768.0

        # Submuestreo para visualización
        if len(audio_data) > max_samples:
            # Tomar promedios de bloques para mantener la forma general
            block_size = len(audio_data) // max_samples
            blocks = len(audio_data) // block_size
            audio_data = audio_data[:blocks * block_size].reshape(blocks, block_size)
            # Tomar el máximo absoluto de cada bloque para mantener picos
            audio_data = np.array([block[np.argmax(np.abs(block))] for block in audio_data])

        return audio_data

    except Exception as e:
        return None


def visualize_segments(audio_file, total_duration, segments, padding_ms):
    """
    Crea una visualización de forma de onda real con segmentos marcados.
    """
    # Extraer datos de forma de onda
    waveform = extract_waveform_data(audio_file, total_duration)

    fig, ax = plt.subplots(figsize=(14, 5))

    if waveform is not None:
        # Crear eje de tiempo
        time_axis = np.linspace(0, total_duration, len(waveform))

        # Dibujar forma de onda base en gris claro
        ax.fill_between(time_axis, waveform, -waveform, color='#CCCCCC', alpha=0.3, linewidth=0)
        ax.plot(time_axis, waveform, color='#888888', linewidth=0.5, alpha=0.5)

        padding_s = padding_ms / 1000.0

        # Crear máscaras para segmentos mantenidos
        for start, end in segments:
            start_with_padding = max(0, start - padding_s)
            end_with_padding = min(total_duration, end + padding_s)

            # Encontrar índices correspondientes
            mask = (time_axis >= start_with_padding) & (time_axis <= end_with_padding)

            # Dibujar segmento mantenido en verde
            ax.fill_between(time_axis[mask], waveform[mask], -waveform[mask],
                          color='#4CAF50', alpha=0.6, linewidth=0)
            ax.plot(time_axis[mask], waveform[mask], color='#2E7D32', linewidth=0.8)

        # Crear lista de segmentos eliminados
        silence_segments = []

        if segments[0][0] > 0.1:
            silence_segments.append((0, segments[0][0]))

        for i in range(len(segments) - 1):
            silence_start = segments[i][1]
            silence_end = segments[i + 1][0]
            if silence_end - silence_start > 0.1:
                silence_segments.append((silence_start, silence_end))

        if segments[-1][1] < total_duration - 0.1:
            silence_segments.append((segments[-1][1], total_duration))

        # Marcar segmentos eliminados con overlay rojo semi-transparente
        for start, end in silence_segments:
            ax.axvspan(start, end, color='#f44336', alpha=0.2)
            # Líneas verticales en los bordes
            ax.axvline(x=start, color='#D32F2F', linewidth=1.5, linestyle='--', alpha=0.7)
            ax.axvline(x=end, color='#D32F2F', linewidth=1.5, linestyle='--', alpha=0.7)

        # Configuración del gráfico
        ax.set_xlim(0, total_duration)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel('Tiempo (segundos)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Amplitud', fontsize=11, fontweight='bold')
        ax.set_title('Forma de Onda del Audio: Verde = Mantenido | Rojo/Sombreado = Eliminado',
                     fontsize=13, fontweight='bold', pad=15)

        # Grid sutil
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        ax.set_facecolor('#F8F9FA')

    else:
        # Fallback: visualización simple si no se puede extraer waveform
        ax.text(0.5, 0.5, 'No se pudo generar la forma de onda\nMostrando vista simplificada',
                ha='center', va='center', fontsize=12, color='#666')
        ax.set_xlim(0, total_duration)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig




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
silence_thresh = st.sidebar.slider(
    "🔇 Umbral de Silencio (dB)",
    min_value=-80,
    max_value=-10,
    value=-40,
    step=5,
    help="Nivel de volumen considerado como silencio. Más negativo = más sensible."
)

# Duración mínima de silencio en milisegundos
min_silence_len = st.sidebar.slider(
    "⏱️ Duración Mínima de Silencio (ms)",
    min_value=100,
    max_value=2000,
    value=500,
    step=50,
    help="Silencios más cortos que esto se mantendrán en el audio."
)

# Margen de seguridad (padding)
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

# ======================== HISTORIAL ========================
st.sidebar.divider()
st.sidebar.header("📜 Historial de Procesamiento")

# Cargar historial desde la base de datos
history_items = load_from_database()

if len(history_items) > 0:
    st.sidebar.markdown(f"**{len(history_items)} archivo(s) procesado(s)**")

    for idx, item in enumerate(history_items):
        with st.sidebar.expander(f"🎵 {item['filename'][:25]}...", expanded=False):
            st.markdown(f"**Procesado:** {item['timestamp']}")
            st.markdown(f"**Duración Original:** {item['original_duration']:.1f}s")
            st.markdown(f"**Duración Final:** {item['cleaned_duration']:.1f}s")
            st.markdown(f"**Tiempo Ahorrado:** {item['time_saved']:.1f}s")

            # Mostrar configuración usada
            with st.expander("⚙️ Configuración usada"):
                st.markdown(f"- Umbral: {item['settings']['umbral_silencio']} dB")
                st.markdown(f"- Duración mín: {item['settings']['duracion_minima']} ms")
                st.markdown(f"- Padding: {item['settings']['padding']} ms")

            # Reproductor de audio
            st.audio(item['audio_data'], format="audio/mp3")

            # Botón de descarga
            download_name = os.path.splitext(item['filename'])[0] + "_limpio.mp3"
            st.download_button(
                label="⬇️ Descargar",
                data=item['audio_data'],
                file_name=download_name,
                mime="audio/mp3",
                key=f"download_history_{idx}"
            )

    # Botón para limpiar historial
    if st.sidebar.button("🗑️ Limpiar Historial", type="secondary"):
        if clear_database():
            st.success("Historial limpiado exitosamente")
            st.rerun()

else:
    st.sidebar.info("No hay archivos procesados aún")


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

    file_extension = uploaded_file.name.split(".")[-1].lower()
    is_video = file_extension in ["mp4", "mov", "mkv"]

    if is_video:
        st.info("🎬 Video detectado. Se extraerá la pista de audio automáticamente.")
    else:
        st.info("🎵 Archivo de audio detectado. Se procesará directamente.")

    if st.button("🚀 Procesar y Limpiar Silencios", type="primary"):
        temp_input = None
        temp_output = None

        try:
            with st.spinner("⏳ Procesando... Esto puede tomar unos momentos."):

                # Guardar archivo subido
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_input:
                    tmp_input.write(uploaded_file.read())
                    temp_input = tmp_input.name

                st.text("📥 Archivo guardado temporalmente...")

                # Obtener duración original
                original_duration = get_audio_duration(temp_input)

                # Detectar silencios
                st.text("🔍 Detectando silencios...")
                nonsilent_segments = detect_silence_with_ffmpeg(
                    temp_input,
                    silence_thresh,
                    min_silence_len
                )

                if not nonsilent_segments:
                    st.warning("⚠️ No se detectaron segmentos de audio.")
                    st.stop()

                st.text(f"✅ Encontrados {len(nonsilent_segments)} segmentos de audio")

                # Crear archivo de salida
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name

                # Unir segmentos
                st.text("✂️ Eliminando silencios y uniendo segmentos...")
                success = merge_segments_with_ffmpeg(
                    temp_input,
                    nonsilent_segments,
                    temp_output,
                    keep_silence
                )

                if not success:
                    st.error("❌ Error al procesar el audio")
                    st.stop()

                # Obtener duración final
                cleaned_duration = get_audio_duration(temp_output)
                time_saved = original_duration - cleaned_duration

                st.success("✅ ¡Procesamiento completado exitosamente!")

                # Estadísticas
                col1, col2, col3 = st.columns(3)
                col1.metric("⏱️ Duración Original", f"{original_duration:.1f}s")
                col2.metric("⏱️ Duración Final", f"{cleaned_duration:.1f}s")
                col3.metric("💨 Tiempo Ahorrado", f"{time_saved:.1f}s")

                st.divider()

                # Visualización de segmentos
                st.subheader("📊 Paso 2: Visualización de Segmentos")
                st.markdown("**Verde** = Audio mantenido | **Rojo** = Silencios eliminados")

                try:
                    fig = visualize_segments(temp_input, original_duration, nonsilent_segments, keep_silence)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"No se pudo generar la visualización: {e}")

                st.divider()

                # Leer archivo de audio procesado
                with open(temp_output, "rb") as audio_file:
                    audio_bytes = audio_file.read()

                # Guardar en base de datos inmediatamente después de procesar
                save_to_database(
                    uploaded_file.name,
                    audio_bytes,
                    original_duration,
                    cleaned_duration,
                    silence_thresh,
                    min_silence_len,
                    keep_silence
                )

                # Reproductor
                st.subheader("🎧 Paso 3: Escucha el resultado")
                st.audio(audio_bytes, format="audio/mp3")

                st.divider()

                # Descarga
                st.subheader("📥 Paso 4: Descarga tu archivo")
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

        except Exception as e:
            st.error(f"❌ Error durante el procesamiento: {str(e)}")
            st.info("""
            **Posibles causas:**
            - El archivo está corrupto o no es válido
            - Formato de archivo no soportado correctamente
            - Problema con ffmpeg

            **Soluciones:**
            - Intenta con otro archivo
            - Verifica que el archivo no esté dañado
            """)

        finally:
            # Limpieza
            if temp_input and os.path.exists(temp_input):
                try:
                    os.unlink(temp_input)
                except:
                    pass

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
        🚀 Desarrollado con Streamlit y ffmpeg |
        💡 Usa ffmpeg directamente para máxima compatibilidad
    </small>
</div>
""", unsafe_allow_html=True)
