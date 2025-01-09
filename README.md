# Proyecto Context Code

## Descripción

**Context Code** es una herramienta que permite generar documentación en formato Markdown a partir de archivos en rutas específicas. Este proyecto utiliza un modelo de inteligencia artificial para interactuar con los usuarios y proporcionar información contextual sobre los archivos analizados.

---

## Funcionalidades de `repo.py`

El archivo `repo.py` ofrece funcionalidades clave para la generación de documentación y el manejo de repositorios de Git. A continuación, se describen sus principales funciones:

### 1. Generación de Markdown a partir de diferencias en Git
- `generate_git_diff_markdown(repo_path: str, ref: str = None) -> str`  
  Genera un resumen en Markdown de las diferencias entre commits, ramas o el último commit de la rama actual.

### 2. Obtención de información de Pull Requests
- `fetch_pull_request(repo: str, pr_number: int) -> dict`  
  Obtiene datos de un Pull Request específico desde la API de GitHub.
- `fetch_pull_request_files(repo: str, pr_number: int) -> list`  
  Obtiene la lista de archivos modificados en un Pull Request.

### 3. Generación de Markdown de Pull Requests
- `generate_markdown_from_pr(repo: str, pr_number: int) -> str`  
  Crea un resumen en formato Markdown que incluye los diffs de los archivos modificados en un Pull Request.

### 4. Generación de Markdown a partir de archivos
- `generate_markdown(paths, ignored_paths=[])`  
  Escanea directorios y archivos para crear un documento Markdown que incluye el contenido de archivos con extensiones específicas.

### 5. Generación de árbol de directorios
- `generate_directory_tree(paths, ignore_dirs=None, max_depth=4)`  
  Genera un árbol de directorios en formato de texto para las rutas especificadas.

---

## Variables de Entorno

Para que el proyecto funcione correctamente, necesitas definir algunas variables de entorno en un archivo `.env` en la raíz del proyecto. A continuación, se detallan las variables necesarias:

- **GPT_MODEL**  
  - **Descripción**: Especifica el modelo de IA que se utilizará para las interacciones.  
  - **Valor por defecto**: `gpt-4o-mini`.

- **OPENAI_API_KEY**  
  - **Descripción**: Tu clave de API de OpenAI, necesaria para autenticarte y realizar llamadas a sus servicios.  
  - **Valor**: `<tu_token_openai>` (debes reemplazarlo con tu clave real).

- **LLM_MONITOR_KEY**  
  - **Descripción**: Clave de monitoreo para el uso del modelo de lenguaje desde https://app.lunary.ai/
  - **Valor**: `<tu_llm_monitor_key>` (opcional).

### Ejemplo de archivo `.env`

```plaintext
GPT_MODEL=gpt-4o-mini
OPENAI_API_KEY=tu_token_openai
LLM_MONITOR_KEY=tu_llm_monitor_key
```

---

## Uso

Para utilizar las funcionalidades de `repo.py`, sigue estos pasos:

1. **Clonar el repositorio**  
   Clona este repositorio o descarga los archivos.

2. **Instalar dependencias**  
   Asegúrate de tener instalado Python 3.7 o superior y las bibliotecas necesarias. Puedes instalar las dependencias usando `pip`:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Ejecutar el script**  
   Puedes ejecutar el script principal y pasarle los argumentos necesarios.

---

### Modos de Uso

#### 1. Generar diferencias de Git
```bash
python repo.py --git <ruta_repo> [rango|rama|commit]
```
- **Ejemplo**: Para generar un resumen de las diferencias entre la rama actual y `main`:
  ```bash
  python repo.py --git /ruta/al/repositorio main
  ```

#### 2. Obtener información de un Pull Request
```bash
python repo.py --pr <repositorio> <número_pr>
```
- **Ejemplo**: Para obtener información sobre un Pull Request específico:
  ```bash
  python repo.py --pr usuario/repositorio 1
  ```

#### 3. Generar Markdown a partir de archivos en directorios
```bash
python repo.py --path <ruta1> <ruta2> ...
```
- **Ejemplo**: Para generar un documento Markdown a partir de archivos en múltiples rutas:
  ```bash
  python repo.py --path /ruta/al/proyecto /otra/ruta
  ```

#### 4. Activar el modo Dev
```bash
python repo.py --dev <ruta>
```
- **Ejemplo**: Para indexar una ruta y poder hacer preguntas con contexto más especificos:
  ```bash
  python repo.py --dev /ruta/al/proyecto
  ```

---

### Ejemplo completo de uso

```bash
# Generar diferencias de Git
python repo.py --git /ruta/al/repositorio

# Obtener información de un Pull Request
python repo.py --pr usuario/repositorio 1

# Generar Markdown a partir de archivos
python repo.py --path /ruta/al/proyecto

# Activar el modo dev
python repo.py --dev /ruta/al/proyecto
```


## TODO
- Agregar razonamiento sobre archivos
- Permitir con técnica ReAct junto a RAG tener más contexto

---

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir, por favor abre un "issue" o un "pull request".

---

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
