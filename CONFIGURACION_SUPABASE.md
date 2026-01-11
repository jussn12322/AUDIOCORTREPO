# 🔧 Configuración de Supabase en Streamlit Cloud

## Paso 1: Editar archivo secrets.toml local

1. Abre el archivo `.streamlit/secrets.toml`
2. Reemplaza `[YOUR-PASSWORD]` con tu password real de Supabase
3. Tu password está en la cadena de conexión que te dio Supabase:
   `postgresql://postgres:TU_PASSWORD_AQUI@db.pyejfpuzjvpdaxpbcrvi.supabase.co:5432/postgres`

## Paso 2: Configurar Streamlit Cloud

1. Ve a tu app en Streamlit Cloud: https://share.streamlit.io/
2. Click en tu app "AUDIOCORTREPO"
3. Click en el menú de 3 puntos (⋮) → **Settings**
4. Ve a la sección **Secrets**
5. Pega exactamente esto (reemplazando [YOUR-PASSWORD] con tu password real):

```toml
[supabase]
host = "aws-0-us-west-1.pooler.supabase.com"
database = "postgres"
user = "postgres.pyejfpuzjvpdaxpbcrvi"
password = "TU_PASSWORD_REAL_AQUI"
port = "6543"
```

**IMPORTANTE:** Usa el puerto **6543** (connection pooling) en lugar de 5432 (conexión directa)

6. Click **Save**
7. La app se redesplegiará automáticamente

## Paso 3: Verificar

1. Espera que la app se redespliega (1-2 minutos)
2. Prueba procesar un archivo
3. Verifica que aparece en el historial
4. Cierra la pestaña y vuelve a abrir
5. El historial debe seguir ahí ✅

## ¿Dónde está mi password?

Tu cadena de conexión completa es:
```
postgresql://postgres:[YOUR-PASSWORD]@db.pyejfpuzjvpdaxpbcrvi.supabase.co:5432/postgres
```

El password es lo que está entre `postgres:` y `@db.pyejfpuzjvpdaxpbcrvi`

## Solución de problemas

### Error: "No se encontraron credenciales"
- Verifica que copiaste los secrets correctamente en Streamlit Cloud
- Asegúrate de hacer click en "Save"
- Redespliega la app manualmente

### Error: "Error conectando a la base de datos"
- Verifica que el password sea correcto
- Verifica que el host sea: `db.pyejfpuzjvpdaxpbcrvi.supabase.co`
- Ve a Supabase y verifica que el proyecto está activo

### El historial no persiste
- Asegúrate de haber configurado los secrets en Streamlit Cloud (no solo local)
- Revisa los logs de la app en Streamlit Cloud
