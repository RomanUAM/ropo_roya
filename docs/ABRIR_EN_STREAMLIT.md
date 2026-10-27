# Abrir la interfaz en Streamlit Community Cloud

Repositorio: `RomanUAM/ropo_roya`. Rama: `main`. Archivo principal: `app.py`.
Utilizar Python 3.12 para coincidir con el entorno de verificación.
`requirements.txt` instala el paquete y las dependencias declaradas en
`pyproject.toml`; no se necesita una clave de OpenAI.

El código ya está preparado, pero el despliegue exige iniciar sesión en la cuenta
de Streamlit del propietario y completar allí la vinculación con GitHub. Guardar
código en GitHub no crea por sí solo una aplicación alojada.

Antes de cargar datos de investigación, configurar el acceso de la aplicación
para el propietario o las personas autorizadas. Los datos y modelos originales
no están en el repositorio. Para el flujo nuevo, comenzar en **Bases nuevas** y
**Entrenar con una base etiquetada**; las versiones históricas necesitan sus
artefactos privados. No se declara un modelo entrenado cuando sólo se ha instalado
la interfaz.

Esta configuración permite instalar y abrir el panel. No sustituye un diseño de
almacenamiento persistente, autenticación y separación de usuarios para un
servicio de producción compartido.
