# Context Code Project

## Description

**Context Code** is a tool designed to generate documentation in Markdown format from files in specific paths. This project uses an artificial intelligence model to interact with users and provide contextual information about the analyzed files.

---

## `repo.py` Functionalities

The `repo.py` file offers key features for generating documentation and managing Git repositories. Below are its main functions:

### 1. Markdown Generation from Git Diffs

- `generate_git_diff_markdown(repo_path: str, ref: str = None) -> str`  
  Generates a Markdown summary of the differences between commits, branches, or the latest commit in the current branch.

### 2. Pull Request Information Retrieval

- `fetch_pull_request(repo: str, pr_number: int) -> dict`  
  Retrieves data of a specific Pull Request from the GitHub API.
- `fetch_pull_request_files(repo: str, pr_number: int) -> list`  
  Gets the list of files modified in a Pull Request.

### 3. Markdown Generation from Pull Requests

- `generate_markdown_from_pr(repo: str, pr_number: int) -> str`  
  Creates a Markdown summary that includes the diffs of the files modified in a Pull Request.

### 4. Markdown Generation from Files

- `generate_markdown(paths, ignored_paths=[])`  
  Scans directories and files to create a Markdown document that includes the content of files with specific extensions.

### 5. Directory Tree Generation

- `generate_directory_tree(paths, ignore_dirs=None, max_depth=4)`  
  Generates a directory tree in text format for the specified paths.

---

## Environment Variables

To ensure the project works correctly, you need to define some environment variables in a `.env` file at the root of the project. Below are the required variables:

- **GPT_MODEL**

  - **Description**: Specifies the AI model to be used for interactions.
  - **Default Value**: `gpt-4o-mini`.

- **OPENAI_API_KEY**

  - **Description**: Your OpenAI API key, required for authentication and service calls.
  - **Value**: `<your_openai_token>` (replace with your actual key).

- **GITHUB_TOKEN**

  - **Description**: Useful for reviewing PRs from GitHub.
  - **Value**: `<your_github_token>`.

- **LLM_MONITOR_KEY**

  - **Description**: Monitoring key for language model usage from https://app.lunary.ai/
  - **Value**: `<your_llm_monitor_key>` (optional).

### Sample `.env` File

```plaintext
GPT_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_token
LLM_MONITOR_KEY=your_llm_monitor_key
```

## Usage

To use the functionalities of repo.py, follow these steps:
	1.	Clone the Repository
Clone this repository or download the files.
	2.	Install Dependencies
Ensure Python 3.7 or higher is installed along with the required libraries. Install the dependencies using pip:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


	3.	Run the Script
Execute the main script and pass the required arguments.

Usage Modes

1. Generate Git Diffs

python repo.py --git <repo_path> [range|branch|commit]

	•	Example: To generate a summary of differences between the current branch and main:

python repo.py --git /path/to/repo main



2. Retrieve Pull Request Information

python repo.py --pr <repository> <pr_number>

	•	Example: To retrieve information about a specific Pull Request:

python repo.py --pr user/repo 1



3. Generate Markdown from Files in Directories

python repo.py --path <path1> <path2> ...

	•	Example: To generate a Markdown document from files in multiple paths:

python repo.py --path /path/to/project /another/path



4. Activate Dev Mode

python repo.py --dev <path>

	•	Example: To index a path and ask more specific context-related questions:

python repo.py --dev /path/to/project

## Full Usage Example

### Generate Git Diffs
python repo.py --git /path/to/repo

### Retrieve Pull Request Information
python repo.py --pr user/repo 1

### Generate Markdown from Files
python repo.py --path /path/to/project

### Activate Dev Mode
python repo.py --dev /path/to/project

## TODO
	•	Add reasoning over files.
	•	Enable ReAct technique alongside RAG for enhanced context.

## Contributions

Contributions are welcome. If you want to contribute, please open an “issue” or a “pull request.”

## License

This project is under the MIT License. See the LICENSE file for more details.
