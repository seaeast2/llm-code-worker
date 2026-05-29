# NVIDIA NIM Code Worker

`nvidia_nim_worker.py` sends a coding task to NVIDIA NIM and prints a unified diff patch.
Codex can handle planning and validation; this script is the code-generation worker. It auto-collects repo context by default so Codex can pass shorter tasks. Pass `--output patch.diff` if you also want the cleaned patch written to a file.

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

If you put both `nvidia-nim-worker` and `nvidia_nim_worker.py` in a directory on your `PATH`, you can run the worker from any folder. The launcher uses the active `VIRTUAL_ENV` Python first, then `.venv` next to the script, and finally `python3` as a fallback:

```bash
nvidia-nim-worker --task "Add a new file named NOTES.txt containing exactly one line: Hello from NIM."
nvidia-nim-worker --task "Add a new file named NOTES.txt containing exactly one line: Hello from NIM." --output patch.diff
```

If the command is not found, add this to your shell profile and open a new terminal:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

On Windows, a `nvidia-nim-worker.cmd` wrapper can be placed alongside `nvidia_nim_worker.py` on `PATH` in the same way.

## Planner

`nvidia_nim_planer.py` runs the worker in a loop, updates `PLAN.md` after each step, and prints the planner's raw LLM response and parsed decision to stderr so you can see what it is doing.

`PLAN.md` is the planner's coordination file. It records:

- the current goal
- the current iteration and phase
- the latest repo status
- the recent progress log

Run it like this:

```bash
python3 nvidia_nim_planer.py --task "hello world 를 출력하는 C 언어 프로그램을 만들어줘."
```

If you want the loop to stop sooner or run longer, use `--max-iterations`.

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
python3 nvidia_nim_worker.py --task "Refactor the parser" --output patch.diff
```
