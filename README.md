# 🎵 AudioCort - Limpiador de Silencios

Aplicación web desarrollada con Streamlit que procesa videos o audios para eliminar silencios automáticamente y exportar el resultado como MP3 de alta calidad.

## 🚀 Características

- **Múltiples formatos soportados:**
  - Videos: MP4, MOV, MKV
  - Audios: MP3, WAV, M4A

- **Procesamiento inteligente:**
  - Extracción automática de audio desde videos
  - Detección configurable de silencios
  - Eliminación de pausas largas con padding suave
  - Exportación a MP3 de alta calidad (192 kbps)

- **Interfaz intuitiva:**
  - Controles deslizantes para ajustar parámetros
  - Vista previa del audio procesado
  - Estadísticas de tiempo ahorrado
  - Descarga directa del archivo limpio

## 📋 Requisitos

- Python 3.8 o superior
- ffmpeg (necesario para procesar videos y múltiples formatos de audio)

## 🔧 Instalación Local

1. Clona este repositorio:
```bash
git clone https://github.com/jussn12322/AUDIOCORTREPO.git
cd AUDIOCORTREPO
```

2. Instala las dependencias de Python:
```bash
pip install -r requirements.txt
```

3. Asegúrate de tener ffmpeg instalado:
   - **Windows:** Descarga desde [ffmpeg.org](https://ffmpeg.org/download.html) y agrega al PATH
   - **macOS:** `brew install ffmpeg`
   - **Linux:** `sudo apt-get install ffmpeg`

4. Ejecuta la aplicación:
```bash
streamlit run app.py
```

## ☁️ Despliegue en Streamlit Cloud

1. Haz fork o clona este repositorio
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu cuenta de GitHub
4. Selecciona este repositorio y `app.py` como archivo principal
5. Streamlit Cloud instalará automáticamente las dependencias de `requirements.txt` y `packages.txt`
6. ¡Listo! Tu app estará disponible en segundos

## ⚙️ Parámetros Configurables

### Umbral de Silencio (dB)
- Rango: -80 dB a -10 dB
- Por defecto: -40 dB
- Valores más negativos detectan más silencios

### Duración Mínima de Silencio (ms)
- Rango: 100 ms a 2000 ms
- Por defecto: 500 ms
- Solo se eliminan silencios que duren al menos este tiempo

### Padding/Margen (ms)
- Rango: 0 ms a 500 ms
- Por defecto: 150 ms
- Mantiene un pequeño margen antes/después de cada segmento para evitar cortes bruscos

## 🎯 Casos de Uso

- Limpiar podcasts grabados con pausas largas
- Procesar clases o conferencias grabadas
- Optimizar audiolibros
- Editar presentaciones grabadas
- Comprimir archivos de audio eliminando silencios

## 🛠️ Tecnologías

- **[Streamlit](https://streamlit.io)** - Framework web para Python
- **[pydub](https://github.com/jiaaro/pydub)** - Manipulación de audio
- **[ffmpeg](https://ffmpeg.org)** - Procesamiento multimedia

## 📝 Estructura del Proyecto

```
AUDIOCORTREPO/
├── app.py              # Aplicación principal de Streamlit
├── requirements.txt    # Dependencias de Python
├── packages.txt        # Dependencias del sistema (ffmpeg)
└── README.md          # Este archivo
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👨‍💻 Autor

Desarrollado por jussn12322

## 🐛 Reporte de Bugs

Si encuentras algún bug o tienes sugerencias, por favor abre un [issue](https://github.com/jussn12322/AUDIOCORTREPO/issues).

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub
