import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_BASE_URL_ENV = "OPENAI_API_BASE"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_ALLOW_MISSING_API_KEY = False
DEFAULT_MODEL = "deepseek-ai/deepseek-v4-flash"
NEMOTRON_NANO_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_REASONING_BUDGET = 0
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_CONTEXT_FILES = ("AGENT.md", "PLAN.md", "README.md", "package.json")
MAX_AUTO_CONTEXT_FILES = 12
ALLOWED_AUTO_CONTEXT_SUFFIXES = {
    ".c",
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".py",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
TASK_STOPWORDS = {
    "add",
    "also",
    "and",
    "build",
    "change",
    "create",
    "code",
    "file",
    "files",
    "from",
    "into",
    "make",
    "need",
    "new",
    "only",
    "patch",
    "please",
    "preserve",
    "refactor",
    "remove",
    "task",
    "the",
    "this",
    "that",
    "update",
    "with",
    "work",
    "write",
    "your",
}
SYSTEM_PROMPT = (
    "You are a coding worker inside a Codex-driven workflow. "
    "Codex will handle planning and validation; your job is to implement the requested change. "
    "Use the provided repository context and infer local conventions from it. "
    "Respond in a terse, technical, unemotional style. Do not use emojis or casual phrasing. "
    "Return exactly one unified diff patch that can be applied with git apply or apply_patch. "
    "The patch must include diff --git headers, --- and +++ file headers, and complete @@ hunk headers. "
    "For a new file, use one hunk starting with @@ -0,0 +1,N @@ and include every added line in that hunk. "
    "Do not include markdown fences, explanations, summaries, or any text outside the patch. "
    "Do not emit a second patch or trailing prose. "
    "Keep the patch minimal, correct, and consistent with the existing code style. "
    "If the task cannot be completed from the provided context, return exactly one line starting with NEEDS_CLARIFICATION: followed by the minimum question."
)


def parse_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if value and value[0] == value[-1] and value[0] in {'\"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


def load_env_files() -> None:
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().with_name(".env")]
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        parse_env_file(candidate)


def missing_key_message() -> str:
    return (
        "OPENAI_API_KEY is not set. Configure it in your environment or a .env file.\n"
        "Linux/macOS examples:\n"
        "  export OPENAI_API_KEY=\"...\"\n"
        "  export OPENAI_API_BASE=\"https://integrate.api.nvidia.com/v1\"\n"
        "Windows PowerShell examples:\n"
        "  $env:OPENAI_API_KEY=\"...\"\n"
        "  $env:OPENAI_API_BASE=\"https://integrate.api.nvidia.com/v1\"\n"
        "Windows CMD examples:\n"
        "  set OPENAI_API_KEY=...\n"
        "  set OPENAI_API_BASE=https://integrate.api.nvidia.com/v1\n"
        "You can also create a .env file with OPENAI_API_KEY and OPENAI_API_BASE."
    )


def build_client(args):
    load_env_files()
    api_key = args.api_key if args.api_key is not None else os.getenv(DEFAULT_API_KEY_ENV)
    if not api_key:
        if args.allow_missing_api_key:
            api_key = "local"
        else:
            raise RuntimeError(missing_key_message())

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is not installed. Activate the project virtual environment or install dependencies before running this script."
        ) from exc

    return OpenAI(
        base_url=args.base_url,
        api_key=api_key,
    )


def read_task(task_arg: str | None) -> str:
    if task_arg:
        return task_arg.strip()

    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data

    raise RuntimeError("No task provided. Pass --task or pipe the task description on stdin.")


def resolve_repo_root(override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return Path.cwd().resolve()

    root = result.stdout.strip()
    return Path(root).resolve() if root else Path.cwd().resolve()


def run_git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return result.stdout


def collect_changed_files(repo_root: Path) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    git_commands = [
        ("diff", "--name-only", "--diff-filter=ACMRTUXB"),
        ("diff", "--name-only", "--cached", "--diff-filter=ACMRTUXB"),
        ("ls-files", "--others", "--exclude-standard"),
    ]

    for command in git_commands:
        output = run_git(repo_root, *command)
        for raw_line in output.splitlines():
            rel = raw_line.strip()
            if not rel:
                continue
            path = (repo_root / rel).resolve()
            if path in seen or not path.exists() or path.is_dir():
                continue
            seen.add(path)
            files.append(path)
    return files


def extract_task_terms(task: str) -> list[str]:
    terms: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9_./-]+", task.lower()):
        term = raw.strip("./-_")
        if len(term) < 4:
            continue
        if term in TASK_STOPWORDS:
            continue
        if term.isdigit():
            continue
        if term not in terms:
            terms.append(term)
    return terms


def collect_task_matches(repo_root: Path, task: str, limit: int) -> list[Path]:
    terms = extract_task_terms(task)
    if not terms or limit <= 0:
        return []

    skip_dirs = {".git", "node_modules", "dist", "build", "coverage", "__pycache__"}
    files: list[Path] = []
    seen: set[Path] = set()

    for path in repo_root.rglob("*"):
        if len(files) >= limit:
            break
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.relative_to(repo_root).parts):
            continue
        if path.suffix.lower() not in ALLOWED_AUTO_CONTEXT_SUFFIXES:
            continue

        resolved = path.resolve()
        if resolved in seen:
            continue

        rel_text = str(path.relative_to(repo_root)).lower()
        name_text = path.name.lower()
        if any(term in rel_text or term in name_text for term in terms):
            seen.add(resolved)
            files.append(path)
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue

        if any(term in content for term in terms):
            seen.add(resolved)
            files.append(path)

    return files


def collect_context_paths(
    task: str,
    explicit_paths: list[str],
    repo_root: Path,
    auto_context: bool,
    max_auto_context_files: int,
) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    manifest: list[str] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            return
        if not path.exists() or path.is_dir():
            return
        seen.add(resolved)
        files.append(path)
        try:
            manifest.append(str(path.resolve().relative_to(repo_root.resolve())))
        except Exception:
            manifest.append(str(path.resolve()))

    if auto_context:
        for name in DEFAULT_CONTEXT_FILES:
            add(repo_root / name)
        for path in collect_changed_files(repo_root):
            add(path)
        for path in collect_task_matches(repo_root, task, max_auto_context_files):
            add(path)

    for raw in explicit_paths:
        add(Path(raw).expanduser())

    return files, manifest


def read_context(paths: list[Path], repo_root: Path) -> str:
    if not paths:
        return ""

    chunks: list[str] = []
    for path in paths:
        try:
            display_path = str(path.resolve().relative_to(repo_root.resolve()))
        except Exception:
            display_path = str(path.resolve())
        if not path.exists():
            raise FileNotFoundError(f"Context file not found: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"Context path is a directory, expected a file: {path}")
        content = path.read_text(encoding="utf-8", errors="replace")
        chunks.append(f"### {display_path}\n```text\n{content}\n```")
    return "\n\n".join(chunks)


def build_messages(task: str, context: str, context_manifest: list[str]) -> list[dict[str, str]]:
    user_parts = ["Task:", task]
    if context_manifest:
        user_parts.extend(["", "Auto-selected context:", "\n".join(f"- {item}" for item in context_manifest)])
    if context:
        user_parts.extend(["", "Context:", context])
    user_parts.extend(
        [
            "",
            "Output requirements:",
            "- Return a unified diff patch only.",
            "- Include full git-style file headers: diff --git, --- and +++.",
            "- Include complete @@ hunk headers; do not return a bare hunk.",
            "- For new files, use one hunk that contains all added lines.",
            "- Make the smallest change that satisfies the task.",
            "- Preserve formatting and existing behavior unless the task requires a change.",
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def is_retryable_api_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    return status_code in {404, 408, 429, 500, 502, 503, 504}


def completion_kwargs(args, messages: list[dict[str, str]], stream: bool) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.model == "deepseek-ai/deepseek-v4-flash":
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": args.reasoning_effort,
            },
        }
    elif args.model == NEMOTRON_NANO_MODEL:
        kwargs["extra_body"] = {
            "reasoning_budget": args.reasoning_budget,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    elif args.reasoning_budget > 0:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_budget": args.reasoning_budget,
        }
    if stream:
        kwargs["stream"] = True
    return kwargs


def emit_completion(completion) -> None:
    output_parts: list[str] = []
    choices = getattr(completion, "choices", None)
    if choices is not None:
        content = choices[0].message.content if choices else ""
        if content:
            output_parts.append(content)
            print(clean_patch_output("".join(output_parts)), end="")
        return

    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            output_parts.append(delta.content)

    print(clean_patch_output("".join(output_parts)), end="")
    sys.stdout.flush()


def clean_patch_output(output: str) -> str:
    text = output.strip()
    text = re.sub(r"^(index [0-9a-f]+\.\.[0-9a-f]+) 100644--- /dev/null$", r"\1\n--- /dev/null", text, flags=re.MULTILINE)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    diff_start = text.find("diff --git ")
    if diff_start > 0:
        text = text[diff_start:]
    fence_start = text.find("\n```")
    if fence_start != -1:
        text = text[:fence_start].rstrip()
    lines = [
        line
        for line in text.splitlines()
        if line.strip() not in {"+\\ No newline at end of file", "\\ No newline at end of file"}
    ]
    normalized_lines: list[str] = []
    in_file_diff = False
    has_new_file_mode = False
    for line in lines:
        if line.startswith("diff --git "):
            in_file_diff = True
            has_new_file_mode = False
            normalized_lines.append(line)
            continue
        if in_file_diff and line.startswith("new file mode "):
            has_new_file_mode = True
        if in_file_diff and line == "--- /dev/null" and not has_new_file_mode:
            normalized_lines.append("new file mode 100644")
            has_new_file_mode = True
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def request_completion(client, args, messages: list[dict[str, str]]):
    last_error: Exception | None = None

    for attempt in range(4):
        try:
            return client.chat.completions.create(**completion_kwargs(args, messages, stream=True))
        except Exception as error:
            last_error = error
            if not is_retryable_api_error(error) or attempt == 3:
                break
            print(
                f"Streaming request failed with {getattr(error, 'status_code', 'error')}; retrying ({attempt + 1}/4)...",
                file=sys.stderr,
            )
            time.sleep(2.0 * (attempt + 1))

    print("Falling back to a non-streaming completion request.", file=sys.stderr)
    for attempt in range(3):
        try:
            return client.chat.completions.create(**completion_kwargs(args, messages, stream=False))
        except Exception as error:
            last_error = error
            if not is_retryable_api_error(error) or attempt == 2:
                break
            print(
                f"Non-streaming request failed with {getattr(error, 'status_code', 'error')}; retrying ({attempt + 1}/3)...",
                file=sys.stderr,
            )
            time.sleep(2.0 * (attempt + 1))

    if last_error is not None:
        raise last_error
    raise RuntimeError("NVIDIA NIM request failed without an exception")


def main() -> int:
    load_env_files()
    parser = argparse.ArgumentParser(
        description="Send a coding task to NVIDIA NIM and receive a patch back.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Codex workflow:
  1. Codex plans and validates the change.
  2. Pass a compact implementation task to this worker with --task.
  3. Let the worker auto-collect repo context when possible.
  4. Add explicit source files with --context when needed.
  5. The worker returns one unified diff patch on stdout.
  6. Codex reviews the patch before applying it.

Examples:
  nvidia-nim-worker --task "Add a parser option for dry-run."
  nvidia-nim-worker --task "Refactor the parser." --context src/app.py --context src/util.py
  nvidia-nim-worker --no-auto-context --task "Patch README.md" --context README.md

Environment:
  OPENAI_API_KEY is required. OPENAI_API_BASE is optional and defaults to NVIDIA NIM.
  .env files in the current folder or next to the script are loaded automatically.""",
    )
    parser.add_argument(
        "--task",
        help="Task description. If omitted, the script reads from stdin.",
    )
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        help="Optional file path to include as context. Repeatable.",
    )
    parser.add_argument(
        "--auto-context",
        dest="auto_context",
        action="store_true",
        default=True,
        help="Automatically include repo docs, changed files, and task-matched files as context.",
    )
    parser.add_argument(
        "--no-auto-context",
        dest="auto_context",
        action="store_false",
        help="Disable automatic context discovery.",
    )
    parser.add_argument(
        "--repo-root",
        help="Optional repository root used for automatic context discovery.",
    )
    parser.add_argument(
        "--max-auto-context-files",
        type=int,
        default=MAX_AUTO_CONTEXT_FILES,
        help=f"Maximum number of task-matched files to add automatically (default: {MAX_AUTO_CONTEXT_FILES}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv(DEFAULT_BASE_URL_ENV, DEFAULT_BASE_URL),
        help=f"OpenAI-compatible API base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. If omitted, OPENAI_API_KEY from the environment or .env is used.",
    )
    parser.add_argument(
        "--allow-missing-api-key",
        action="store_true",
        default=DEFAULT_ALLOW_MISSING_API_KEY,
        help="Use a placeholder API key for local OpenAI-compatible servers that do not require authentication.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum number of output tokens (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature for code generation (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Nucleus sampling value (default: {DEFAULT_TOP_P}).",
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=DEFAULT_REASONING_BUDGET,
        help="Reasoning budget passed through to the NVIDIA endpoint.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=("low", "medium", "high"),
        help=f"Reasoning effort for supported NVIDIA endpoints (default: {DEFAULT_REASONING_EFFORT}).",
    )
    args = parser.parse_args()

    task = read_task(args.task)
    repo_root = resolve_repo_root(args.repo_root)
    context_paths, context_manifest = collect_context_paths(
        task=task,
        explicit_paths=args.context,
        repo_root=repo_root,
        auto_context=args.auto_context,
        max_auto_context_files=args.max_auto_context_files,
    )
    context = read_context(context_paths, repo_root)
    messages = build_messages(task, context, context_manifest)

    client = build_client(args)
    completion = request_completion(client, args, messages)
    emit_completion(completion)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
