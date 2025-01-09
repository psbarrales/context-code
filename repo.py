import os
import sqlite3
import sys
import fnmatch
import requests
import tiktoken
from git import Repo, GitCommandError
from colorama import Fore, Style, init
from dotenv import load_dotenv
from typing import Annotated
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage

from rich.console import Console
from rich.markdown import Markdown

from langchain_chroma import Chroma
from langchain_community.callbacks import LLMonitorCallbackHandler
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

# Importa la clase Document
from langchain.schema import Document
from typing_extensions import TypedDict
import hashlib

init(autoreset=True)
load_dotenv()
console = Console()

# Token de GitHub para analizar Pull Requests
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://api.github.com"

if not GITHUB_TOKEN:
    print(Fore.YELLOW + "Advertencia: No se encontró GITHUB_TOKEN. Las funcionalidades de Pull Request pueden fallar.")


def generate_git_diff_markdown(repo_path: str, ref: str = None) -> str:
    """
    Genera un resumen en Markdown con las diferencias de un commit, rango de commits,
    rama contra 'main', o el último commit de la rama actual si no se proporciona ref.
    """
    repo = Repo(repo_path)
    markdown_parts = []

    try:
        # Si no se pasa referencia, usar el último commit de la rama actual
        if not ref:
            ref = repo.head.commit.hexsha
            commit = repo.commit(ref)
            if len(commit.parents) > 0:
                parent = commit.parents[0]
                diff = repo.git.diff(parent.hexsha, commit.hexsha)
                markdown_parts.append(
                    f"# Cambios en el último commit {ref} (vs {parent.hexsha})")
            else:
                # Si no tiene padres, es el primer commit
                diff = repo.git.show(commit.hexsha)
                markdown_parts.append(
                    f"# Cambios en el último commit {ref} (Primer commit)")
        elif "..." in ref:  # Rango de commits
            base, target = ref.split("...")
            diff = repo.git.diff(base, target)
            markdown_parts.append(f"# Cambios entre {base} y {target}")
        elif len(ref) == 40 or repo.commit(ref):  # Un commit SHA o nombre de referencia válido
            commit = repo.commit(ref)
            if len(commit.parents) > 0:
                parent = commit.parents[0]
                diff = repo.git.diff(parent.hexsha, commit.hexsha)
                markdown_parts.append(
                    f"# Cambios en el commit {ref} (vs {parent.hexsha})")
            else:
                # Si no tiene padres, es el primer commit
                diff = repo.git.show(commit.hexsha)
                markdown_parts.append(
                    f"# Cambios en el commit {ref} (Primer commit)")
        else:  # Rama comparada con main
            main_branch = "main"
            diff = repo.git.diff(main_branch, ref)
            markdown_parts.append(f"# Cambios entre {ref} y {main_branch}")

        markdown_parts.append("```diff")
        markdown_parts.append(diff)
        markdown_parts.append("```")
    except GitCommandError as e:
        markdown_parts.append(f"Error: {e}")

    return "\n".join(markdown_parts)


def fetch_pull_request(repo: str, pr_number: int) -> dict:
    """
    Obtiene los datos de un Pull Request desde la API de GitHub.
    """
    url = f"{BASE_URL}/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(
            Fore.RED + f"Error: No se pudo obtener el Pull Request. Código de estado: {response.status_code}")
        print(Fore.YELLOW +
              f"Mensaje: {response.json().get('message', 'No se encontró mensaje')}")
        sys.exit(1)

    return response.json()


def fetch_pull_request_files(repo: str, pr_number: int) -> list:
    """
    Obtiene la lista de archivos modificados en un Pull Request desde la API de GitHub.
    """
    url = f"{BASE_URL}/repos/{repo}/pulls/{pr_number}/files"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(
            Fore.RED + f"Error: No se pudo obtener los archivos del Pull Request. Código de estado: {response.status_code}")
        print(Fore.YELLOW +
              f"Mensaje: {response.json().get('message', 'No se encontró mensaje')}")
        sys.exit(1)

    return response.json()


def generate_markdown_from_pr(repo: str, pr_number: int) -> str:
    """
    Genera un resumen en formato Markdown con los diffs de un Pull Request.
    """
    pr_data = fetch_pull_request(repo, pr_number)
    files_data = fetch_pull_request_files(repo, pr_number)

    title = pr_data.get("title", "Sin título")
    number = pr_data.get("number", "Desconocido")
    author = pr_data.get("user", {}).get("login", "Desconocido")
    created_at = pr_data.get("created_at", "Desconocido")
    body = pr_data.get("body", "Sin descripción")

    markdown = [
        f"# Pull Request #{number}: {title}",
        f"**Autor:** {author}",
        f"**Creado en:** {created_at}",
        "",
        "## Descripción",
        f"{body}",
        "",
        "## Cambios propuestos",
    ]

    # Agregar diffs de los archivos modificados
    for file in files_data:
        filename = file.get("filename", "Desconocido")
        patch = file.get("patch", "No hay cambios visibles.")

        markdown.append(f"### Archivo: `{filename}`")
        markdown.append("```diff")
        markdown.append(patch)
        markdown.append("```")
        markdown.append("")  # Línea en blanco para separar los diffs

    return "\n".join(markdown)


def generate_markdown(paths, ignored_paths=[]):
    """
    Recursively iterates over each of the paths in 'paths', searching for
    files with specific extensions and excluding common folders and wildcard patterns.
    Dynamically excludes files and folders based on 'ignored_paths'.
    Returns a string with the content in Markdown format.
    """
    markdown_parts = []

    # Folders and wildcard patterns to ignore
    ignore_folders = {".venv", "node_modules", "__pycache__", ".git"}
    ignore_patterns = ["*.sample", ".*"]

    # Add dynamically provided ignored paths to ignore_folders
    ignore_folders.update(os.path.basename(path)
                          for path in ignored_paths if os.path.isdir(path))

    # Print the paths being processed
    print(Fore.BLUE + "Processing the following paths:")
    files_add = []

    for path in paths:
        print(Fore.GREEN + f"- {os.path.abspath(path)}")

    for base_path in paths:
        base_path = os.path.abspath(base_path)

        for root, dirs, files in os.walk(base_path):
            # Skip ignored directories
            if any(os.path.commonpath([root, ignored_path]) == ignored_path for ignored_path in ignored_paths):
                continue

            # Filter out folders to ignore
            dirs[:] = [d for d in dirs if d not in ignore_folders]

            for filename in files:
                # Check for ignored patterns
                if any(fnmatch.fnmatch(filename, pattern) for pattern in ignore_patterns):
                    continue  # Skip files matching any ignore pattern

                # Supported extensions
                supported_extensions = {
                    '.ts': 'ts',
                    '.tsx': 'tsx',
                    '.js': 'javascript',
                    '.jsx': 'javascript',
                    '.py': 'python',
                    '.json': 'json',
                    '.md': 'markdown'
                }

                # Detect the file extension
                _, extension = os.path.splitext(filename)
                if extension in supported_extensions:
                    files_add.append(filename)
                    language = supported_extensions[extension]
                    full_path = os.path.join(root, filename)

                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except Exception as e:
                        content = f"Error reading the file {full_path}: {e}"

                    markdown_parts.append(f"## {full_path}")
                    markdown_parts.append(f"```{language}")
                    markdown_parts.append(content)
                    markdown_parts.append("```")
                    markdown_parts.append("")  # Optional blank line

    print(Fore.CYAN + "Files added:", files_add)
    return "\n".join(markdown_parts)


def generate_directory_tree(paths, ignore_dirs=None, max_depth=4):
    """
    Genera un árbol de directorios en formato de texto para múltiples rutas.
    Combina las carpetas predeterminadas a ignorar con las proporcionadas.
    """
    # Carpetas ignoradas por defecto
    default_ignore_dirs = {
        '.git', 'node_modules', '__pycache__', 'android', 'ios',
        'build', 'dist', 'venv', '.idea', '.vscode', 'fonts', '.venv'
    }

    # Combinar carpetas ignoradas predeterminadas con las adicionales
    combined_ignore_dirs = default_ignore_dirs.union(set(ignore_dirs or []))

    structure = []

    # Print the paths being processed
    print(Fore.BLUE + "Processing the following paths:")

    for path in paths:
        print(Fore.GREEN + f"- {os.path.abspath(path)}")

    # Si paths es una lista, iterar sobre cada ruta
    if isinstance(paths, list):
        for path in paths:
            structure.append(f"Source path: {path}")
            structure.append(_generate_tree_for_path(
                path, combined_ignore_dirs, max_depth))
    else:
        # Manejar un solo path como string
        structure.append(_generate_tree_for_path(
            paths, combined_ignore_dirs, max_depth))

    return '\n'.join(["```"] + structure + ["```"])


def _generate_tree_for_path(start_path, ignore_dirs, max_depth):
    """
    Genera el árbol de directorios para una sola ruta.
    """
    structure = []
    base_level = start_path.rstrip(os.sep).count(os.sep)

    for root, dirs, files in os.walk(start_path):
        # Filtrar directorios ignorados
        dirs[:] = [
            d for d in dirs
            if d not in ignore_dirs and not d.startswith('.')
        ]

        current_level = root.rstrip(os.sep).count(os.sep) - base_level
        if current_level >= max_depth:
            # No descender más allá del nivel máximo
            dirs[:] = []
            continue

        indent = '    ' * current_level
        folder = os.path.basename(root)
        structure.append(f"{indent}├── {folder}/")

        subindent = '    ' * (current_level + 1)
        for f in files:
            structure.append(f"{subindent}├── {f}")

    return '\n'.join(structure)


"""
AI Section
"""


class State(TypedDict):
    messages: Annotated[list, add_messages]


class AIBot:
    model_name = os.getenv("GPT_MODEL", "gpt-4o-mini")
    monitor = LLMonitorCallbackHandler(app_id=os.getenv("LLM_MONITOR_KEY", ""))

    def __init__(self, hash):
        self.hash = hash
        print('Starting Bot: ' + hash)
        self.llm = ChatOpenAI(model=self.model_name, callbacks=[self.monitor])
        self.config = {"configurable": {"thread_id": hash}}
        sqlite3_conn = sqlite3.connect(
            f"data/memory_{hash}.sqlite", check_same_thread=False)

        self.memory = SqliteSaver(sqlite3_conn)

    def set_context(self, context: str):
        # self.routes = routes
        self.context = context

    def chatbot(self, state: State):
        """Invoca al LLM con las 'messages' previas."""
        ROUTES_CONTEXT = (
            f"Tienes conocimiento sobre lo siguiente\n"
            "Úsalas como referencia para ayudar al usuario en todo lo relacionado con estos archivos.\n"
            "Contenido archivos:\n"
            f"{self.context}\n"
        )
        response = self.llm.invoke(
            [SystemMessage(ROUTES_CONTEXT)] + state["messages"])
        return {"messages": [response]}

    def compile(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        self.graph = graph_builder.compile(checkpointer=self.memory)

    def stream_graph_updates(self, user_input: str, first_message: bool = False):
        """
        Envía el input del usuario al grafo y recoge la respuesta generada por la IA.
        - first_message: si es True, incluye un mensaje de sistema con el contexto de rutas.
        """
        # Si es la primera vez que hablamos con la IA, inyectamos un 'system' con el contexto
        # ROUTES_CONTEXT = (
        #     f"Tienes conocimiento sobre lo siguiente\n"
        #     "Úsalas como referencia para ayudar al usuario en todo lo relacionado con estos archivos.\n"
        #     "Contenido archivos:\n"
        #     f"{self.context}\n"
        # )
        messages_to_send = []
        # if first_message:
        #     messages_to_send.append(("system", ROUTES_CONTEXT))

        messages_to_send.append(("user", user_input))

        for event in self.graph.stream({"messages": messages_to_send}, self.config):
            # Iteramos sobre cada "evento" que produce el grafo
            for value in event.values():
                assistant_message = value["messages"][-1].content
                md_text = Fore.LIGHTRED_EX + "Assistant:\n" + \
                    Fore.RESET + assistant_message
                md = Markdown(md_text)
                console.print(md)


def read_multiline_input(start_line: str = ""):
    print(Fore.LIGHTCYAN_EX +
          "Pegue su contenido y, cuando termine, escriba `\"\"\"` (en una línea) para cerrar:")
    lines = []
    if len(start_line) > 0 and start_line != "\"\"\"":
        lines.append(start_line)
    while True:
        line = input()
        if line.strip().endswith('"""'):
            break
        lines.append(line)
    return Fore.LIGHTYELLOW_EX + "\n".join(lines)


def num_tokens_from_string(string: str, encoding_name: str = "gpt-4o-mini") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def index_every_file(paths, vectorstore: Chroma):
    """
    Itera sobre cada archivo en las rutas dadas y genera un solo embedding
    por path (un documento por ruta).

    - No valida si cambió o no el contenido. Simplemente indexa el archivo.
    - Usa la ruta completa (full_path) como 'id' del documento.
    """
    ignore_folders = {
        '.git', 'node_modules', '__pycache__', 'android', 'ios',
        'build', 'dist', 'venv', '.idea', '.vscode', 'fonts', '.venv', 'locales', 'i18n', 'locale', 'assets'
    }
    ignore_patterns = ["*.sample", ".*", "*mock*"]
    supported_extensions = {
        '.ts': 'ts',
        '.tsx': 'tsx',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.py': 'python',
        '.json': 'json',
        '.md': 'markdown'
    }

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=num_tokens_from_string,
        is_separator_regex=False,
    )
    for base_path in paths:
        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if d not in ignore_folders]
            for filename in files:
                if any(fnmatch.fnmatch(filename, pattern) for pattern in ignore_patterns):
                    continue

                _, extension = os.path.splitext(filename)
                if extension in supported_extensions:
                    full_path = os.path.join(root, filename)
                    try:
                        loader = TextLoader(full_path)
                        docs = loader.load()

                        combined_content = "\n".join(
                            d.page_content for d in docs
                        )

                        docs = text_splitter.create_documents(
                            [combined_content])

                        print(f'Docs count {full_path} {len(docs)}')

                        for doc in docs:
                            doc.id = full_path
                            doc.page_content = f"{full_path}\n```...\n{doc.page_content}\n...```"
                            doc.metadata = {
                                "path": full_path
                            }
                            vectorstore.add_documents([doc])

                    except Exception as e:
                        print(f"Error procesando {full_path}: {e}")


class DevBotAI:
    model_name = os.getenv("GPT_MODEL", "gpt-4o-mini")
    monitor = LLMonitorCallbackHandler(app_id=os.getenv("LLM_MONITOR_KEY", ""))

    def __init__(self, hash, store: Chroma):
        self.hash = hash
        self.store = store
        print('Starting Bot: ' + hash)
        self.llm = ChatOpenAI(model=self.model_name,
                              temperature=0.4,
                              callbacks=[self.monitor])
        self.config = {"configurable": {"thread_id": hash}}
        sqlite3_conn = sqlite3.connect(
            f"data/memory_{hash}.sqlite", check_same_thread=False)

        self.memory = SqliteSaver(sqlite3_conn)

    def set_context(self, context: str):
        # self.routes = routes
        self.context = context

    def chatbot(self, state: State):
        """Invoca al LLM con las 'messages' previas."""
        store = self.store
        ROUTES_CONTEXT = (
            f"Eres un agente developer capaz de investigar, estudiar, buscar y analizar en un repositorio que el usuario indique.\n"
            "Úsalas como referencia para ayudar al usuario en todo lo relacionado con estos archivos.\n"
            "# Contexto\n"
            f"{self.context}\n\n"
            "### Importante\n"
            "No te apresures a responder\n"
            "Asegura que tienes el contexto completo para responder\n"
            "Si necesitas más cosas del usuario puedes preguntar\n"
        )
        items = store.search(
            query=state["messages"][-1].content, search_type="similarity", k=20)
        memories = "\n\n".join(
            item.page_content for item in items)
        memories = f"## Relevantes:\n{memories}" if memories else ""
        response = self.llm.invoke(
            [SystemMessage(ROUTES_CONTEXT), SystemMessage(memories)] + state["messages"])
        return {"messages": [response]}

    def chatbot_no_memory(self, state: State):
        """Invoca al LLM con las 'messages' previas."""
        ROUTES_CONTEXT = (
            f"Eres un agente developer capaz de investigar, estudiar, buscar y analizar en un repositorio que el usuario indique.\n"
            "Úsalas como referencia para ayudar al usuario en todo lo relacionado con estos archivos.\n"
            "# Contexto\n"
            f"{self.context}\n\n"
            "### Importante\n"
            "No te apresures a responder\n"
            "Asegura que tienes el contexto completo para responder\n"
            "Si necesitas más cosas del usuario puedes preguntar\n"
        )
        response = self.llm.invoke(
            [SystemMessage(ROUTES_CONTEXT)] + state["messages"])
        return {"messages": [response]}

    def memory_prompt(self, state: State):
        """Invoca al LLM con las 'messages' previas."""
        store = self.store
        ROUTES_CONTEXT = (
            f"Eres un agente developer capaz de investigar, estudiar, buscar y analizar en un repositorio que el usuario indique.\n"
            "Úsalas como referencia para ayudar al usuario en todo lo relacionado con estos archivos.\n"
            "# Contexto\n"
            f"{self.context}\n\n"
            "### Importante\n"
            "No te apresures a responder\n"
            "Asegura que tienes el contexto completo para responder\n"
            "Si necesitas más cosas del usuario puedes preguntar\n"
        )
        items = store.search(
            query=state["messages"][-1].content, search_type="similarity", k=20)
        memories = "\n\n".join(
            item.page_content for item in items)
        memories = f"## Relevantes:\n{memories}" if memories else ""
        return [SystemMessage(ROUTES_CONTEXT), SystemMessage(memories)] + state["messages"]

    def compile(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        # self.graph = graph_builder.compile(
        #     checkpointer=self.memory)
        self.graph = create_react_agent(
            self.llm, tools=[], checkpointer=self.memory, state_modifier=self.memory_prompt)

        graph_builder_no_memory = StateGraph(State)
        graph_builder_no_memory.add_node("chatbot", self.chatbot_no_memory)
        graph_builder_no_memory.add_edge(START, "chatbot")
        graph_builder_no_memory.add_edge("chatbot", END)
        self.graph_no_memory = graph_builder_no_memory.compile(
            checkpointer=self.memory)

    def greeting(self):
        messages_to_send = []
        messages_to_send.append(
            ("system", "Saluda al usuario, invitalo a usarte"))

        for event in self.graph_no_memory.stream({"messages": messages_to_send}, self.config):
            # Iteramos sobre cada "evento" que produce el grafo
            for value in event.values():
                assistant_message = value["messages"][-1].content
                md_text = Fore.LIGHTRED_EX + "Assistant:\n" + \
                    Fore.RESET + assistant_message
                md = Markdown(md_text)
                # console.print(md)

    def stream_graph_updates(self, user_input: str, first_message: bool = False):
        """
        Envía el input del usuario al grafo y recoge la respuesta generada por la IA.
        - first_message: si es True, incluye un mensaje de sistema con el contexto de rutas.
        """
        messages_to_send = []
        messages_to_send.append(("user", user_input))

        for event in self.graph.stream({"messages": messages_to_send}, self.config):
            # Iteramos sobre cada "evento" que produce el grafo
            for value in event.values():
                assistant_message = value["messages"][-1].content
                md_text = Fore.LIGHTRED_EX + "Assistant:\n" + \
                    Fore.RESET + assistant_message
                md = Markdown(md_text)
                console.print(md)


def developer(valid_paths, ignored_paths=[]):

    greeting = [" ▄▄▄  ▗▞▀▚▖▄   ▄     ▗▖  ▗▖ ▄▄▄     ▐▌▗▞▀▚▖",
                "▐▌  █ ▐▛▀▀▘█   █     ▐▛▚▞▜▌█   █    ▐▌▐▛▀▀▘",
                "▐▌  █ ▝▚▄▄▖ ▀▄▀      ▐▌  ▐▌▀▄▄▄▀ ▗▞▀▜▌▝▚▄▄▖",
                "▐▙▄▄▀                ▐▌  ▐▌      ▝▚▄▟▌      \n\n"]
    print("\n".join(greeting))

    hash = hashlib.md5("-".join(valid_paths).encode() +
                       "-".join(ignored_paths).encode() +
                       "-".join("developer").encode()).hexdigest()
    hash_index = hashlib.md5("-".join(valid_paths).encode() +
                             "-".join(ignored_paths).encode()).hexdigest()
    print(Fore.LIGHTRED_EX + f"Hash Index: {hash_index}")
    print(Fore.LIGHTRED_EX + f"Hash: {hash}")
    tree = generate_directory_tree(valid_paths, ignored_paths)
    embedding = OpenAIEmbeddings(model='text-embedding-3-small')

    # Determinar el directorio de persistencia
    persist_directory = f"./data/data_{hash_index}"

    should_index = True
    # Verificar si ya existe la carpeta en persist_directory
    if os.path.exists(persist_directory):
        print(
            f"La carpeta de persistencia '{persist_directory}' ya existe. "
            "Omitiendo proceso de indexación."
        )
        should_index = False

    # Inicializar el vectorstore (Chroma) solo si la carpeta NO existe
    vectorstore = Chroma(
        collection_name=f"data_{hash_index}",
        embedding_function=embedding,
        persist_directory=persist_directory
    )

    if should_index:
        # Indexar todos los archivos
        print(
            "Proceso de indexación iniciado"
        )
        index_every_file(valid_paths, vectorstore)

    bot = DevBotAI(hash, vectorstore)
    bot.set_context(tree)
    bot.compile()
    bot.greeting()

    user_input_loop(bot)


def user_input_loop(bot: AIBot):
    first_time = True
    memories = bot.memory.list(bot.config, limit=1)
    for memory in memories:
        messages = memory.checkpoint['channel_values']['messages']
        for message in messages:
            type_class = message.__class__.__name__
            if type_class != 'SystemMessage':
                if type_class == 'AIMessage':
                    md_text = Fore.LIGHTRED_EX + "Assistant: \n" + \
                        Fore.RESET + message.content + '\n\n'
                    md = Markdown(md_text)
                    console.print(md)
                else:
                    md_text = Fore.CYAN + "\n\n" + "-"*40 + Fore.CYAN + "\n\nTú: " + \
                        Fore.RESET + Fore.LIGHTYELLOW_EX + message.content
                    md = Markdown(md_text)
                    console.print(md)
    while True:
        try:
            user_input = input(Fore.CYAN + "\n" + "-"*40 +
                               "\nTú: " + Fore.RESET + Fore.LIGHTYELLOW_EX)
            content = user_input
            if user_input.startswith('"""'):  # Si empieza con tres comillas
                content = read_multiline_input(user_input)
            elif user_input.lower() in ["exit", "quit"]:
                print("Saliendo...")
                break
            bot.stream_graph_updates(content, first_time)
            first_time = False  # A partir de aquí, ya no enviamos el system message cada vez
        except Exception as e:
            # Fallback si input() falla (por ejemplo en algunos entornos)
            # user_input = "¿Qué sabes sobre las rutas definidas?"
            # print("User:", user_input)
            # ai_bot.stream_graph_updates(user_input, first_time)
            print("Saliendo...", e)
            break
    pass


def main():
    """
    Main function to handle different modes (--git, --pr, or --path).
    """
    if len(sys.argv) < 2:
        print(Fore.RED + "Uso: python script.py --git|--pr|--path [opciones]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "--dev":
        if len(sys.argv) < 3:
            print(Fore.RED + "Uso: python script.py --dev <ruta1> <ruta2> ...")
            sys.exit(1)
        paths = sys.argv[2:]
        # Separar rutas válidas de las ignoradas
        valid_paths = []
        ignored_paths = []
        for path in paths:
            if path.startswith('-'):
                # Quitar el prefijo '-' de las paths ignoradas
                ignored_paths.append(path[1:])
            else:
                valid_paths.append(path)
        developer(valid_paths, ignored_paths)

        return

    elif mode == "--git":
        if len(sys.argv) < 3:
            print(
                Fore.RED + "Uso: python script.py --git <ruta_repo> [rango|rama|commit]")
            sys.exit(1)
        repo_path = sys.argv[2]
        ref = sys.argv[3] if len(sys.argv) > 3 else None
        markdown = generate_git_diff_markdown(repo_path, ref)
    elif mode == "--pr":
        if len(sys.argv) < 4:
            print(Fore.RED + "Uso: python script.py --pr <repositorio> <número_pr>")
            sys.exit(1)
        repo = sys.argv[2]
        pr_number = int(sys.argv[3])
        # pr_data = fetch_pull_request(repo, pr_number)
        markdown = generate_markdown_from_pr(repo, pr_number)
    elif mode == "--path":
        if len(sys.argv) < 3:
            print(Fore.RED + "Uso: python script.py --path <ruta1> <ruta2> ...")
            sys.exit(1)
        paths = sys.argv[2:]
        # Separar rutas válidas de las ignoradas
        valid_paths = []
        ignored_paths = []
        for path in paths:
            if path.startswith('-'):
                # Quitar el prefijo '-' de las paths ignoradas
                ignored_paths.append(path[1:])
            else:
                valid_paths.append(path)
        tree = generate_directory_tree(valid_paths, ignored_paths)
        markdown = generate_markdown(valid_paths, ignored_paths)
        markdown = (
            f"{tree}\n\n"
            f"{markdown}"
        )

    else:
        print(Fore.RED + "Modo no reconocido. Usa --git, --pr, --path o --dev")
        sys.exit(1)

    print(Fore.GREEN + "Markdown generado exitosamente:")
    print(Style.RESET_ALL + markdown)

    hash = hashlib.md5("-".join(sys.argv[2:]).encode()).hexdigest()
    ai_bot = AIBot(hash)

    ai_bot.set_context(markdown)
    ai_bot.compile()

    tokens_count = num_tokens_from_string(markdown, "gpt-4o-mini")

    print('Total tokens: ', tokens_count)

    # for message in ai_bot.memory.list(ai_bot.config):
    #     print('message', message)

    user_input_loop(ai_bot)


if __name__ == "__main__":
    main()
