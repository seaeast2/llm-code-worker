# NVIDIA NIM Code Worker

`nvidia_nim_worker.py` sends a coding task to NVIDIA NIM and prints a unified diff patch.
Codex can handle planning and validation; this script is the code-generation worker. It auto-collects repo context by default so Codex can pass shorter tasks.

## Setup

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Fill in your API key in `.env`:

```env
OPENAI_API_KEY=your_nvidia_nim_api_key
OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
```

## Run

Linux/macOS:

```bash
python3 nvidia_nim_worker.py --task "Add a new file named NOTES.txt containing exactly one line: Hello from NIM."
```

Windows PowerShell:

```powershell
python nvidia_nim_worker.py --task "Add a new file named NOTES.txt containing exactly one line: Hello from NIM."
```

Windows CMD:

```cmd
python nvidia_nim_worker.py --task "Add a new file named NOTES.txt containing exactly one line: Hello from NIM."
```


## Global command

If `~/.local/bin` is on your `PATH`, you can run the worker from any folder:

```bash
nvidia-nim-worker --task "Add a new file named NOTES.txt containing exactly one line: Hello from NIM."
```

If the command is not found, add this to your shell profile and open a new terminal:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

On Windows, `local-llm-worker.cmd` can be placed on `PATH` in the same way as `nvidia-nim-worker.cmd`.

## Local LLM worker

`local_llm_worker.py` reuses the same patch-generation workflow with a local OpenAI-compatible server:

```text
Base URL: http://192.168.0.19:8080
Model: qwen2.5-coder-7b-instruct-q6_k
API key: not required
```

Run it like the NVIDIA worker:

```bash
python local_llm_worker.py --task "Add a parser option for dry-run."
local-llm-worker --task "Add a parser option for dry-run."
```

## Context files

The worker auto-collects repository docs, changed files, and task-matched files by default. Pass one or more files with `--context` when you want to force extra context, or disable automatic discovery with `--no-auto-context`:

```bash
python3 nvidia_nim_worker.py --task "Refactor the parser" --context path/to/file.py --context path/to/another.py
python3 nvidia_nim_worker.py --no-auto-context --task "Refactor the parser" --context path/to/file.py
```
