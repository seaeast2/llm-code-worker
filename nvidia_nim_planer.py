import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import difflib
import re

import nvidia_nim_worker

DEFAULT_PLANNER_MODEL = "deepseek-ai/deepseek-v4-flash"
DEFAULT_MAX_ITERATIONS = 8
PLAN_FILENAME = "PLAN.md"
PLAN_TEMPLATE = """# Autonomous Plan

This file is managed automatically by `nvidia_nim_planer.py`.

## Goal
{goal}

## State
- Repo root: {repo_root}
- Iteration: {iteration}/{max_iterations}
- Phase: {phase}
- Status: {status}{current_task_block}{latest_summary_block}

## Repo Status
{repo_status}

## Progress Log
{progress_log}
"""

PLANNER_SYSTEM_PROMPT = """You are a planning agent for an autonomous 3-stage coding loop.
The loop repeats until the goal is done or the iteration limit is reached.
You are responsible for choosing exactly one next worker task at a time.
Use the repository status and the current PLAN.md contents as the source of truth.
Treat PLAN.md as a coordination file managed by the planner; ignore the fact that it is usually modified.

Return strictly valid JSON only:
{
  "action": "run_worker" | "done",
  "worker_task": "string",
  "done_reason": "string"
}

Rules:
- If action is "run_worker", worker_task must be short, specific, and implementation-focused.
- If action is "done", done_reason must briefly explain why the goal is already satisfied.
- Never include markdown fences or any text outside JSON.
"""


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def git_status_short(repo_root: Path) -> str:
    code, out, _ = run_cmd(["git", "status", "--short"], repo_root)
    if code != 0:
        return ""
    lines = [line for line in out.splitlines() if PLAN_FILENAME not in line]
    return "\n".join(lines).strip()


def read_task(task_arg: str | None) -> str:
    if task_arg:
        return task_arg.strip()
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data
    try:
        data = input("Planner goal> ").strip()
    except EOFError as exc:
        raise RuntimeError(
            "No task provided. Pass --task, pipe the task description on stdin, or type a goal interactively."
        ) from exc
    if data:
        return data
    raise RuntimeError(
        "No task provided. Pass --task, pipe the task description on stdin, or type a goal interactively."
    )


def truncate(text: str, limit: int = 320) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def plan_file_path(repo_root: Path) -> Path:
    return repo_root / PLAN_FILENAME


def format_history_line(record: dict[str, str]) -> str:
    parts: list[str] = [f"iteration {record['iteration']}"]
    if record.get("planner_task"):
        parts.append(f"task={truncate(record['planner_task'], 120)}")
    if record.get("worker_result"):
        parts.append(f"worker={truncate(record['worker_result'], 120)}")
    if record.get("apply_result"):
        parts.append(f"apply={truncate(record['apply_result'], 120)}")
    if record.get("verify_result"):
        parts.append(f"verify={truncate(record['verify_result'], 120)}")
    if record.get("decision"):
        parts.append(f"decision={truncate(record['decision'], 120)}")
    return "; ".join(parts)


def render_plan_md(
    user_goal: str,
    repo_root: Path,
    max_iterations: int,
    iteration: int,
    phase: str,
    status: str,
    repo_status: str,
    current_task: str = "",
    latest_summary: str = "",
    history: list[dict[str, str]] | None = None,
) -> str:
    history = history or []
    current_task_block = f"\n- Current task: {current_task}" if current_task else ""
    latest_summary_block = f"\n- Latest summary: {latest_summary}" if latest_summary else ""
    if repo_status:
        repo_status_block = f"```text\n{repo_status}\n```"
    else:
        repo_status_block = "(clean)"
    if history:
        progress_log = "\n".join(f"- {format_history_line(record)}" for record in history[-10:])
    else:
        progress_log = "- (none yet)"
    return PLAN_TEMPLATE.format(
        goal=user_goal.strip() or "(empty)",
        repo_root=repo_root,
        iteration=iteration,
        max_iterations=max_iterations,
        phase=phase,
        status=status,
        current_task_block=current_task_block,
        latest_summary_block=latest_summary_block,
        repo_status=repo_status_block,
        progress_log=progress_log,
    )


def write_plan_md(
    repo_root: Path,
    user_goal: str,
    max_iterations: int,
    iteration: int,
    phase: str,
    status: str,
    repo_status: str,
    current_task: str = "",
    latest_summary: str = "",
    history: list[dict[str, str]] | None = None,
) -> Path:
    path = plan_file_path(repo_root)
    path.write_text(
        render_plan_md(
            user_goal=user_goal,
            repo_root=repo_root,
            max_iterations=max_iterations,
            iteration=iteration,
            phase=phase,
            status=status,
            repo_status=repo_status,
            current_task=current_task,
            latest_summary=latest_summary,
            history=history,
        ),
        encoding="utf-8",
    )
    return path


def read_plan_md(repo_root: Path) -> str:
    path = plan_file_path(repo_root)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def build_planner_messages(user_goal: str, repo_status: str, plan_md: str) -> list[dict[str, str]]:
    lines: list[str] = [f"User goal:\n{user_goal}"]
    lines.append(
        "PLAN.md (authoritative coordination state):\n"
        + (plan_md.strip() if plan_md.strip() else "(missing)")
    )
    if repo_status:
        lines.append(f"Current git status (short, PLAN.md ignored):\n{repo_status}")
    else:
        lines.append("Current git status (short, PLAN.md ignored):\n(clean)")
    lines.append("Decide the next single action now.")
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(lines)},
    ]


def planner_step(client, args, messages: list[dict[str, str]], user_goal: str | None = None, repo_root: Path | None = None) -> tuple[dict[str, str], str]:
    # If no remote client is available, use a simple local rule-based planner for basic tasks
    if client is None:
        # Attempt to parse simple delete/remove instructions
        if user_goal and is_delete_instruction(user_goal):
            tokens = extract_file_tokens(user_goal)
            if tokens:
                worker_task = f"Delete the file or directory named '{tokens[0]}' in the repository root {repo_root or '(unknown)'}"
                parsed = {"action": "run_worker", "worker_task": worker_task}
                return parsed, json.dumps(parsed, ensure_ascii=False)
        # Fallback: declare done (no LLM available to plan)
        parsed = {"action": "done", "done_reason": "No LLM client available and no simple rule matched."}
        return parsed, json.dumps(parsed, ensure_ascii=False)

    response = client.chat.completions.create(
        model=args.planner_model,
        messages=messages,
        temperature=args.planner_temperature,
        top_p=args.planner_top_p,
        max_tokens=args.planner_max_tokens,
    )
    content = response.choices[0].message.content.strip() if response.choices else ""
    if not content:
        raise RuntimeError("Planner returned empty output.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Planner output is not valid JSON:\n{content}") from exc
    action = parsed.get("action")
    if action not in {"run_worker", "done"}:
        raise RuntimeError(f"Planner returned invalid action: {action}")
    if action == "run_worker" and not parsed.get("worker_task"):
        raise RuntimeError("Planner returned run_worker without worker_task.")
    if action == "done" and not parsed.get("done_reason"):
        parsed["done_reason"] = "Planner declared completion."
    return parsed, content


def run_worker(worker_script: str, worker_task: str, repo_root: Path, allow_missing_api_key: bool = False) -> tuple[int, str, str]:
    worker_path = Path(worker_script)
    if worker_path.suffix == ".py":
        cmd = [sys.executable, worker_script, "--task", worker_task]
    else:
        cmd = [worker_script, "--task", worker_task]
    if allow_missing_api_key:
        cmd.append("--allow-missing-api-key")
    return run_cmd(cmd, repo_root)


def apply_patch(repo_root: Path, patch_text: str) -> tuple[bool, str, list[str]]:
    """Try to apply a git-style patch. Supports two worker output formats:
    1) standard unified git diff (preferred), or
    2) an explicit machine-parsable file block format produced when diffs are hard.

    The file-block format (exact) is:
    ===FILE: <path> ACTION: <create|modify|delete>===
    ```text
    <file contents>
    ```

    Returns (success, message, changed_files_list).
    """
    # First, check for explicit file-block format (machine-parsable)
    file_block_re = re.compile(
        r"^===FILE: (?P<path>.+?) ACTION: (?P<action>create|modify|delete)===(?:\r?\n(?:```(?:text)?\r?\n(?P<content>.*?)(?:\r?\n```))?)?",
        re.M | re.S,
    )
    blocks = list(file_block_re.finditer(patch_text))
    if blocks:
        applied: list[str] = []
        failures: list[str] = []
        for b in blocks:
            path = b.group("path").strip()
            action = b.group("action").strip()
            content = b.group("content") or ""
            target = Path(repo_root) / path
            try:
                if action in ("create", "modify"):
                    if not target.parent.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                    code, out, err = run_cmd(["git", "add", "--", path], repo_root)
                    if code != 0:
                        failures.append(f"git add failed for {path}: {err or out}")
                        continue
                    applied.append(path)
                elif action == "delete":
                    if target.exists():
                        code, out, err = run_cmd(["git", "rm", "--", path], repo_root)
                        if code != 0:
                            failures.append(f"git rm failed for {path}: {err or out}")
                            continue
                    applied.append(path)
            except Exception as exc:
                failures.append(f"Exception applying block for {path}: {exc}")
        if failures:
            return False, "Errors applying file blocks:\n" + "\n".join(failures), []

        # Commit applied changes (if any)
        if applied:
            commit_msg = f"Apply file blocks: {', '.join(applied)}\n\nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
            code, out, err = run_cmd(["git", "commit", "-m", commit_msg, "--no-verify"], repo_root)
            if code != 0:
                return True, f"Applied file blocks; files changed: {', '.join(applied)} (git commit failed: {err or out})", applied
            return True, f"Applied file blocks and committed: {', '.join(applied)}", applied

        return True, "Applied file blocks; no files changed.", []

    # Otherwise attempt to treat worker output as a git unified diff
    patch = nvidia_nim_worker.clean_patch_output(patch_text)
    if patch and not patch.endswith("\n"):
        patch += "\n"
    if "diff --git " not in patch:
        return False, "Worker output did not include a git patch.", []

    # Quick validation for deleted files
    deleted_paths: list[str] = []
    diff_iter = list(re.finditer(r"diff --git a/([^ ]+) b/([^\\n]+)", patch))
    for idx, m in enumerate(diff_iter):
        start = m.start()
        end = diff_iter[idx + 1].start() if idx + 1 < len(diff_iter) else len(patch)
        chunk = patch[start:end]
        if "deleted file mode" in chunk:
            a_path = m.group(1)
            deleted_paths.append(a_path)

    if deleted_paths:
        missing: list[str] = []
        for p in deleted_paths:
            if not (repo_root / p).exists():
                code, _, _ = run_cmd(["git", "ls-files", "--error-unmatch", p], repo_root)
                if code != 0:
                    missing.append(p)
        if missing:
            # Suggest likely candidates
            code, ls_out, _ = run_cmd(["git", "ls-files"], repo_root)
            candidates = ls_out.splitlines() if code == 0 else []
            for root, _, files in os.walk(repo_root):
                if ".git" in root.split(os.sep):
                    continue
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), repo_root)
                    if rel not in candidates:
                        candidates.append(rel)

            suggestions: dict[str, str] = {}
            for m in missing:
                base = os.path.basename(m)
                starts = [c for c in candidates if os.path.basename(c).startswith(base)]
                if len(starts) == 1:
                    suggestions[m] = starts[0]
                    continue
                basenames = [os.path.basename(c) for c in candidates]
                close = difflib.get_close_matches(base, basenames, n=3, cutoff=0.6)
                mapped: list[str] = []
                for c in close:
                    for cand in candidates:
                        if os.path.basename(cand) == c:
                            mapped.append(cand)
                            break
                if mapped:
                    suggestions[m] = mapped[0]

            msg_lines = [f"Missing files in patch: {', '.join(missing)}"]
            if suggestions:
                sug_text = ", ".join(f"{k} -> {v}" for k, v in suggestions.items())
                msg_lines.append(f"Suggested matches: {sug_text}")
            msg_lines.append("Worker produced a deletion for a file that does not exist. Ask the planner to re-run or fix the worker task.")
            return False, "\n".join(msg_lines), []

    # First try the fast path with git apply
    check = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=repo_root,
        input=patch,
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        apply_run = subprocess.run(
            ["git", "apply", "-"],
            cwd=repo_root,
            input=patch,
            capture_output=True,
            text=True,
        )
        if apply_run.returncode == 0:
            # determine changed files from the patch
            file_re = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+?)", re.M)
            changed_files: list[str] = [m.group("b") for m in file_re.finditer(patch)]

            # Stage changed files intelligently
            stage_failures: list[str] = []
            for f in changed_files:
                target = Path(repo_root) / f
                if target.exists():
                    code, out, err = run_cmd(["git", "add", "--", f], repo_root)
                else:
                    # file likely deleted; update index for that path
                    code, out, err = run_cmd(["git", "add", "-u", "--", f], repo_root)
                if code != 0:
                    stage_failures.append(f)

            # Commit staged changes
            if changed_files:
                commit_msg = f"Apply patch from worker: {', '.join(changed_files)}\n\nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
                code, out, err = run_cmd(["git", "commit", "-m", commit_msg, "--no-verify"], repo_root)
                if code != 0:
                    return True, f"Patch applied; files changed: {', '.join(changed_files)} (git commit failed: {err or out})", changed_files
                return True, f"Patch applied and committed: {', '.join(changed_files)}", changed_files

            return True, "Patch applied.", []
        return False, f"git apply failed:\n{apply_run.stderr.strip() or apply_run.stdout.strip()}", []

    # If git apply check failed, attempt robust fallback parsing and applying
    stderr_msg = check.stderr.strip() or check.stdout.strip()

    try:
        # Pre-normalize some common malformed outputs (e.g., '+++ b/foo@@')
        patch = re.sub(r"(\+\+\+ b/[^\n]+)@@", r"\1\n@@", patch)
        patch = re.sub(r"(--- a/[^\n]+)@@", r"\1\n@@", patch)

        # Split into per-file diff blocks
        file_re = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+?)\n(?P<body>.*?)(?=^diff --git a/|\Z)", re.M | re.S)
        applied_files: list[str] = []
        failures: list[str] = []

        for m in file_re.finditer(patch):
            a_path = m.group("a")
            b_path = m.group("b")
            body = m.group("body") or ""
            body_lines = body.splitlines()

            is_new = "--- /dev/null" in body or "new file mode" in body or "new file\n" in body or "\nnew file\n" in body or "\nnew file " in body or "new file" in body
            is_delete = "+++ /dev/null" in body or "deleted file mode" in body

            target = Path(repo_root) / b_path

            if is_delete:
                # delete file if exists
                if target.exists():
                    code, out, err = run_cmd(["git", "rm", "--", str(b_path)], repo_root)
                    if code != 0:
                        failures.append(f"git rm failed for {b_path}: {err or out}")
                        continue
                    applied_files.append(b_path)
                else:
                    # already missing; record as applied
                    applied_files.append(b_path)
                continue

            if is_new:
                # collect added lines (lines starting with '+', but not '+++')
                content_lines: list[str] = []
                for line in body_lines:
                    if line.startswith("+") and not line.startswith("+++"):
                        content_lines.append(line[1:])
                # write file
                if not target.parent.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("\n".join(content_lines) + ("\n" if content_lines and not content_lines[-1].endswith("\n") else ""), encoding="utf-8")
                # stage
                code, out, err = run_cmd(["git", "add", "--", str(b_path)], repo_root)
                if code != 0:
                    failures.append(f"git add failed for {b_path}: {err or out}")
                    continue
                applied_files.append(b_path)
                continue

            # Otherwise treat as a modification: apply hunks manually
            # Find hunks
            hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.M)
            lines = body.splitlines()
            i = 0
            hunks: list[dict] = []
            while i < len(lines):
                if lines[i].startswith("@@ "):
                    match = hunk_re.match(lines[i])
                    if not match:
                        i += 1
                        continue
                    old_start = int(match.group(1))
                    old_count = int(match.group(2) or "1")
                    new_start = int(match.group(3))
                    new_count = int(match.group(4) or "1")
                    i += 1
                    hunk_lines: list[str] = []
                    while i < len(lines) and not lines[i].startswith("@@ ") and not lines[i].startswith("diff --git "):
                        hunk_lines.append(lines[i])
                        i += 1
                    hunks.append({
                        "old_start": old_start,
                        "old_count": old_count,
                        "new_start": new_start,
                        "new_count": new_count,
                        "lines": hunk_lines,
                    })
                else:
                    i += 1

            if not hunks:
                failures.append(f"No hunks found for modification of {b_path}")
                continue

            # read original file
            if not target.exists():
                failures.append(f"Target file for modification not found: {b_path}")
                continue
            orig_text = target.read_text(encoding="utf-8", errors="replace")
            orig_lines = orig_text.splitlines()

            new_lines: list[str] = []
            cur_idx = 0  # 0-based index into orig_lines
            ok = True
            for h in hunks:
                old_start = h["old_start"] - 1
                # append unchanged lines before this hunk
                if cur_idx < old_start:
                    new_lines.extend(orig_lines[cur_idx:old_start])
                    cur_idx = old_start
                # apply hunk
                for hl in h["lines"]:
                    if hl.startswith(" "):
                        # context
                        val = hl[1:]
                        if cur_idx >= len(orig_lines) or orig_lines[cur_idx] != val:
                            ok = False
                            break
                        new_lines.append(val)
                        cur_idx += 1
                    elif hl.startswith("-"):
                        val = hl[1:]
                        if cur_idx >= len(orig_lines) or orig_lines[cur_idx] != val:
                            ok = False
                            break
                        cur_idx += 1
                    elif hl.startswith("+"):
                        new_lines.append(hl[1:])
                    else:
                        # unknown line type, treat as context
                        new_lines.append(hl)
                if not ok:
                    break
            if not ok:
                failures.append(f"Hunk application failed for {b_path}")
                continue
            # append the rest
            if cur_idx < len(orig_lines):
                new_lines.extend(orig_lines[cur_idx:])

            # write new content
            if not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(new_lines) + ("\n" if new_lines and not new_lines[-1].endswith("\n") else ""), encoding="utf-8")
            code, out, err = run_cmd(["git", "add", "--", str(b_path)], repo_root)
            if code != 0:
                failures.append(f"git add failed for {b_path}: {err or out}")
                continue
            applied_files.append(b_path)

        if failures:
            return False, f"Fallback apply had errors:\n" + "\n".join(failures) + ("\nOriginal git apply error:\n" + stderr_msg if stderr_msg else ""), []

        # Commit applied_files
        if applied_files:
            commit_msg = f"Patch applied via fallback: {', '.join(applied_files)}\n\nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
            code, out, err = run_cmd(["git", "commit", "-m", commit_msg, "--no-verify"], repo_root)
            if code != 0:
                return True, f"Patch applied via fallback; files changed: {', '.join(applied_files)} (git commit failed: {err or out})", applied_files
            return True, f"Patch applied via fallback and committed; files changed: {', '.join(applied_files)}", applied_files

        return True, "Patch applied via fallback; no files changed.", []

    except Exception as exc:
        return False, f"git apply --check failed:\n{stderr_msg}\nFallback exception: {exc}", []


# Helpers to sanitize planner worker_task values and discover repository candidates

def _list_repo_candidates(repo_root: Path) -> list[str]:
    code, ls_out, _ = run_cmd(["git", "ls-files"], repo_root)
    candidates: list[str] = []
    if code == 0:
        candidates = [l.strip() for l in ls_out.splitlines() if l.strip()]
    # include files from working tree (untracked)
    for root, _, files in os.walk(repo_root):
        if ".git" in root.split(os.sep):
            continue
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), repo_root)
            if rel not in candidates:
                candidates.append(rel)
    return candidates


def ensure_git_repo(repo_root: Path) -> None:
    """Ensure the directory is a git repository and has basic user config set.

    - If git is not present, do nothing and warn.
    - If the directory is not a git repo, run `git init`.
    - Ensure local git config has user.name and user.email set (from env or defaults).
    """
    if shutil.which("git") is None:
        print("[planner] Warning: git not found on PATH; git operations will be skipped.", file=sys.stderr)
        return

    # Is it already a git repo?
    code, out, err = run_cmd(["git", "rev-parse", "--is-inside-work-tree"], repo_root)
    if code != 0:
        print(f"[planner] Initializing git repository at {repo_root}", file=sys.stderr)
        code2, out2, err2 = run_cmd(["git", "init"], repo_root)
        if code2 != 0:
            print(f"[planner] git init failed: {err2 or out2}", file=sys.stderr)
            return
        # set a sensible default branch name
        run_cmd(["git", "config", "init.defaultBranch", "main"], repo_root)

    # Ensure user.name is set locally
    code, name_out, _ = run_cmd(["git", "config", "--get", "user.name"], repo_root)
    if code != 0 or not (name_out or "").strip():
        name = os.getenv("GIT_AUTHOR_NAME") or os.getenv("GIT_COMMITTER_NAME") or "nvidia-nim-planner"
        run_cmd(["git", "config", "user.name", name], repo_root)
        print(f"[planner] git config user.name set to {name}", file=sys.stderr)

    # Ensure user.email is set locally
    code, email_out, _ = run_cmd(["git", "config", "--get", "user.email"], repo_root)
    if code != 0 or not (email_out or "").strip():
        email = os.getenv("GIT_AUTHOR_EMAIL") or os.getenv("GIT_COMMITTER_EMAIL") or f"noreply@{os.uname().nodename}"
        run_cmd(["git", "config", "user.email", email], repo_root)
        print(f"[planner] git config user.email set to {email}", file=sys.stderr)


def sanitize_worker_task(repo_root: Path, task: str) -> tuple[str, list[tuple[str, str]]]:

    """Try to detect filenames mentioned in the planner task and replace truncated/ambiguous names
    with best matching repository paths when there is a clear single candidate.

    Returns (possibly_modified_task, list_of_replacements).
    """
    replacements: list[tuple[str, str]] = []
    candidates = _list_repo_candidates(repo_root)

    # 1) quoted tokens
    tokens: list[str] = re.findall(r"['\"]([^'\"]+)['\"]", task)
    # 2) file-like tokens (has an extension)
    if not tokens:
        tokens.extend(re.findall(r"([^\s,]+?\.[A-Za-z0-9]+)", task))
    # 3) tokens after explicit markers like 'named' or 'called' (avoid matching 'file' alone in phrases like 'file or directory')
    tokens.extend(re.findall(r"(?:named|called)\s+['\"]?([^'\"\s,()]+)['\"]?", task))

    # dedupe preserving order
    seen: set[str] = set()
    dedup_tokens: list[str] = []
    for t in tokens:
        if t and t not in seen:
            seen.add(t)
            dedup_tokens.append(t)

    for tkn in dedup_tokens:
        t = tkn.strip()
        if not t:
            continue
        # exists as-is?
        if (repo_root / t).exists() or t in candidates:
            continue
        base = os.path.basename(t)
        matches = [c for c in candidates if os.path.basename(c) == base]
        if not matches:
            matches = [c for c in candidates if base in os.path.basename(c) or os.path.basename(c).startswith(base)]
        if not matches:
            basenames = [os.path.basename(c) for c in candidates]
            close = difflib.get_close_matches(base, basenames, n=3, cutoff=0.6)
            mapped: list[str] = []
            for c in close:
                for cand in candidates:
                    if os.path.basename(cand) == c:
                        mapped.append(cand)
                        break
            matches = mapped
        if len(matches) == 1:
            new = matches[0]
            task = task.replace(tkn, new)
            replacements.append((tkn, new))
        # ambiguous or none -> skip and let planner re-run
    return task, replacements


def extract_file_tokens(text: str) -> list[str]:
    """Return list of file-like tokens found in freeform text (e.g., 'hello.c')."""
    tokens = re.findall(r"([^\s,()\"'<>]+?\.[A-Za-z0-9]+)", text or "")
    return [t.strip() for t in tokens if t.strip()]


def is_delete_instruction(text: str) -> bool:
    """Rudimentary check for delete/remove intent in English and Korean."""
    if not text:
        return False
    t = text.lower()
    keywords = ["delete", "remove", "rm", "삭제", "지워", "제거", "삭제해", "삭제해주세요", "삭제해줘"]
    return any(k in t for k in keywords)


def perform_delete_files(repo_root: Path, tokens: list[str]) -> tuple[bool, str]:
    """Resolve tokens to repository paths and delete them (tracked: git rm + commit, untracked: remove file).
    Returns (success, message).
    """
    if not tokens:
        return False, "No file tokens provided."
    candidates = _list_repo_candidates(repo_root)
    resolved: list[str] = []
    not_found: list[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        candidate_path = (repo_root / tok)
        if candidate_path.exists():
            rel = os.path.relpath(str(candidate_path), str(repo_root))
            resolved.append(rel)
            continue
        base = os.path.basename(tok)
        matches = [c for c in candidates if os.path.basename(c) == base]
        if not matches:
            matches = [c for c in candidates if base in os.path.basename(c) or os.path.basename(c).startswith(base)]
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        not_found.append(tok)
    if not_found:
        return False, f"Could not find unique file(s) for: {', '.join(not_found)}"

    tracked: list[str] = []
    untracked: list[str] = []
    for p in resolved:
        code, _, _ = run_cmd(["git", "ls-files", "--error-unmatch", p], repo_root)
        if code == 0:
            tracked.append(p)
        else:
            if (repo_root / p).exists():
                untracked.append(p)
            else:
                return False, f"File not found: {p}"

    msgs: list[str] = []
    if tracked:
        cmd = ["git", "rm", "--"] + tracked
        code, out, err = run_cmd(cmd, repo_root)
        if code != 0:
            return False, f"git rm failed: {err or out}"
        files_str = ", ".join(tracked)
        commit_msg = f"Remove {files_str}\n\nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
        code, out, err = run_cmd(["git", "commit", "-m", commit_msg, "--no-verify"], repo_root)
        if code != 0:
            msgs.append(f"git commit failed: {err or out}")
        else:
            msgs.append(f"Committed removal of {files_str}")

    for p in untracked:
        try:
            (repo_root / p).unlink()
            msgs.append(f"Removed untracked file {p}")
        except Exception as exc:
            return False, f"Failed to remove {p}: {exc}"

    return True, "; ".join(msgs) if msgs else "No files changed."


def run_verify_commands(repo_root: Path, commands: list[str]) -> tuple[bool, str]:
    if not commands:
        return True, "No verify command configured."
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            shell=True,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            return False, f"Verify failed: {cmd}\nstdout:\n{out}\nstderr:\n{err}"
    return True, "Verify commands passed."


def print_block(title: str, content: str) -> None:
    print(f"[planner] {title}:", file=sys.stderr)
    if content:
        print(content, file=sys.stderr)
    else:
        print("(empty)", file=sys.stderr)


def main() -> int:
    nvidia_nim_worker.load_env_files()
    parser = argparse.ArgumentParser(
        description="Autonomous planner that keeps a PLAN.md file updated while looping planner -> worker -> apply/verify.",
    )
    parser.add_argument("--task", help="Top-level user goal. If omitted, read from stdin.")
    parser.add_argument("--repo-root", help="Repository root. Defaults to git root or cwd.")
    parser.add_argument("--worker-script", default="nvidia-nim-worker", help="Worker command or script path.")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--verify-cmd", action="append", default=[], help="Post-apply verification command.")
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument(
        "--base-url",
        default=os.getenv(nvidia_nim_worker.DEFAULT_BASE_URL_ENV, nvidia_nim_worker.DEFAULT_BASE_URL),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--allow-missing-api-key", action="store_true", default=False)
    parser.add_argument("--planner-max-tokens", type=int, default=4096)
    parser.add_argument("--planner-temperature", type=float, default=0.2)
    parser.add_argument("--planner-top-p", type=float, default=0.9)
    args = parser.parse_args()

    user_goal = read_task(args.task)
    repo_root = nvidia_nim_worker.resolve_repo_root(args.repo_root)
    worker_script = args.worker_script.strip()
    worker_label = worker_script
    plan_path = plan_file_path(repo_root)

    worker_path = Path(worker_script)
    if worker_path.suffix == ".py" or worker_path.is_absolute() or any(sep in worker_script for sep in ("/", os.sep)):
        worker_path = worker_path.expanduser()
        if not worker_path.is_absolute():
            worker_path = (Path.cwd() / worker_path).resolve()
        if not worker_path.exists():
            raise RuntimeError(f"Worker script not found: {worker_path}")
        worker_script = str(worker_path)
        worker_label = worker_script
    elif shutil.which(worker_script) is None:
        raise RuntimeError(f"Worker command not found on PATH: {worker_script}")

    # Ensure git repository and minimal config for repo-local git operations
    ensure_git_repo(repo_root)

    # Quick local execution path for simple filesystem tasks (delete/remove files).
    file_tokens = extract_file_tokens(user_goal)
    if file_tokens and is_delete_instruction(user_goal):
        ok, msg = perform_delete_files(repo_root, file_tokens)
        if ok:
            print(f"[planner] Direct action succeeded: {msg}", file=sys.stderr)
            return 0
        else:
            print(f"[planner] Direct action failed: {msg}", file=sys.stderr)

    try:
        client = nvidia_nim_worker.build_client(args)
    except Exception as exc:
        print(f"[planner] Warning: failed to build LLM client: {exc}", file=sys.stderr)
        client = None
    history: list[dict[str, str]] = []

    print(f"[planner] repo_root={repo_root}", file=sys.stderr)
    print(f"[planner] worker_script={worker_label}", file=sys.stderr)
    print(f"[planner] plan_file={plan_path}", file=sys.stderr)
    print(f"[planner] max_iterations={args.max_iterations}", file=sys.stderr)
    print("[planner] flow=plan -> worker -> apply/verify -> repeat", file=sys.stderr)

    for iteration in range(1, args.max_iterations + 1):
        repo_status = git_status_short(repo_root)
        plan_md = read_plan_md(repo_root)
        if not plan_md:
            write_plan_md(
                repo_root=repo_root,
                user_goal=user_goal,
                max_iterations=args.max_iterations,
                iteration=iteration,
                phase="planning",
                status="initializing",
                repo_status=repo_status,
                history=history,
            )
            plan_md = read_plan_md(repo_root)

        write_plan_md(
            repo_root=repo_root,
            user_goal=user_goal,
            max_iterations=args.max_iterations,
            iteration=iteration,
            phase="planning",
            status="asking planner for the next task",
            repo_status=repo_status,
            history=history,
        )
        plan_md = read_plan_md(repo_root)
        messages = build_planner_messages(user_goal=user_goal, repo_status=repo_status, plan_md=plan_md)

        print(f"[planner] stage 1/3: planning (iteration {iteration}/{args.max_iterations})", file=sys.stderr)
        decision, planner_raw = planner_step(client, args, messages, user_goal=user_goal, repo_root=repo_root)
        print_block("planner raw response", planner_raw)
        print_block("planner parsed response", json.dumps(decision, indent=2, ensure_ascii=False))

        if decision["action"] == "done":
            done_reason = decision.get("done_reason", "")
            history.append(
                {
                    "iteration": str(iteration),
                    "decision": f"done: {done_reason}",
                    "apply_result": "not needed",
                    "verify_result": "not needed",
                }
            )
            write_plan_md(
                repo_root=repo_root,
                user_goal=user_goal,
                max_iterations=args.max_iterations,
                iteration=iteration,
                phase="done",
                status=f"planner declared completion: {done_reason}",
                repo_status=repo_status,
                latest_summary=done_reason,
                history=history,
            )
            print(f"[planner] done: {done_reason}", file=sys.stderr)
            return 0

        planner_task = decision["worker_task"].strip()
        write_plan_md(
            repo_root=repo_root,
            user_goal=user_goal,
            max_iterations=args.max_iterations,
            iteration=iteration,
            phase="worker",
            status="worker running",
            repo_status=repo_status,
            current_task=planner_task,
            latest_summary="planner chose the next worker task",
            history=history,
        )

        print(f"[planner] stage 2/3: worker task -> {planner_task}", file=sys.stderr)
        sanitized_task, replacements = sanitize_worker_task(repo_root, planner_task)
        if replacements:
            repl_text = ", ".join(f"'{old}' -> '{new}'" for old, new in replacements)
            print_block("planner adjusted worker task", repl_text)
            print_block("planner used task", sanitized_task)

        # If the planner requested a simple delete, perform it locally (no worker needed)
        if is_delete_instruction(sanitized_task):
            toks = extract_file_tokens(sanitized_task)
            if toks:
                ok, msg = perform_delete_files(repo_root, toks)
                print_block("planner performed local delete", msg)
                if ok:
                    history.append(
                        {
                            "iteration": str(iteration),
                            "planner_task": planner_task,
                            "worker_result": "local_delete",
                            "apply_result": "local_delete",
                            "verify_result": "not attempted",
                            "decision": "local delete executed",
                        }
                    )
                    write_plan_md(
                        repo_root=repo_root,
                        user_goal=user_goal,
                        max_iterations=args.max_iterations,
                        iteration=iteration,
                        phase="done",
                        status=f"local delete executed: {toks}",
                        repo_status=git_status_short(repo_root),
                        latest_summary=msg,
                        history=history,
                    )
                    print(f"[planner] completed local delete: {msg}", file=sys.stderr)
                    return 0
                else:
                    print(f"[planner] local delete failed: {msg}", file=sys.stderr)
                    # fallthrough to worker

        # Encourage the worker to be decisive and provide a machine-parsable output
        wrapper = (
            "Do not ask for clarification. If important details are missing, make reasonable choices.\n"
            "Prefer a unified git diff patch. If that is difficult, return explicit file blocks ONLY using the exact format below:\n"
            "===FILE: <path> ACTION: <create|modify|delete>===\n```text\n<file contents>\n```\n"
            "When using the file-block format, do not include any other text.\n"
            "If returning a unified diff, follow standard git diff format (diff --git, --- a/..., +++ b/..., @@ hunks)."
        )
        used_task = wrapper + "\n" + sanitized_task
        code, worker_out, worker_err = run_worker(worker_script, used_task, repo_root, allow_missing_api_key=args.allow_missing_api_key)
        worker_out = (worker_out or "").strip()
        worker_err = (worker_err or "").strip()
        print_block("worker stdout", worker_out)
        print_block("worker stderr", worker_err)
        print(f"[planner] worker exit={code}", file=sys.stderr)

        if code != 0:
            history.append(
                {
                    "iteration": str(iteration),
                    "planner_task": planner_task,
                    "worker_result": f"exit={code}",
                    "apply_result": "not attempted",
                    "verify_result": "not attempted",
                    "decision": "worker failed",
                }
            )
            write_plan_md(
                repo_root=repo_root,
                user_goal=user_goal,
                max_iterations=args.max_iterations,
                iteration=iteration,
                phase="planning",
                status="worker failed; planner will retry in the next iteration",
                repo_status=repo_status,
                current_task=planner_task,
                latest_summary=truncate(worker_err or worker_out or f"exit={code}"),
                history=history,
            )
            continue

        print("[planner] stage 3/3: apply/verify", file=sys.stderr)
        applied, apply_msg, changed_files = apply_patch(repo_root, worker_out)
        print(f"[planner] apply result: {apply_msg}", file=sys.stderr)
        if not applied:
            history.append(
                {
                    "iteration": str(iteration),
                    "planner_task": planner_task,
                    "worker_result": "exit=0",
                    "apply_result": apply_msg,
                    "verify_result": "not attempted",
                    "decision": "apply failed",
                }
            )
            write_plan_md(
                repo_root=repo_root,
                user_goal=user_goal,
                max_iterations=args.max_iterations,
                iteration=iteration,
                phase="planning",
                status="apply failed; planner will retry in the next iteration",
                repo_status=git_status_short(repo_root),
                current_task=planner_task,
                latest_summary=apply_msg,
                history=history,
            )
            continue

        verify_ok, verify_msg = run_verify_commands(repo_root, args.verify_cmd)
        print(f"[planner] verify result: {verify_msg}", file=sys.stderr)
        history.append(
            {
                "iteration": str(iteration),
                "planner_task": planner_task,
                "worker_result": "exit=0",
                "apply_result": apply_msg,
                "verify_result": verify_msg,
                "decision": "verify passed" if verify_ok else "verify failed",
            }
        )
        write_plan_md(
            repo_root=repo_root,
            user_goal=user_goal,
            max_iterations=args.max_iterations,
            iteration=iteration,
            phase="planning",
            status=(
                "iteration completed; planner will continue"
                if verify_ok
                else "verify failed; planner will retry in the next iteration"
            ),
            repo_status=git_status_short(repo_root),
            current_task=planner_task,
            latest_summary=verify_msg,
            history=history,
        )

        if verify_ok:
            print("[planner] iteration completed successfully", file=sys.stderr)

            # If the user's top-level goal was to delete file(s), verify completion locally and finish early
            if is_delete_instruction(user_goal):
                goal_tokens = extract_file_tokens(user_goal)
                if goal_tokens:
                    all_missing = True
                    candidates = _list_repo_candidates(repo_root)
                    for t in goal_tokens:
                        # check working tree directly and candidate list
                        if (repo_root / t).exists() or any(os.path.basename(c) == os.path.basename(t) for c in candidates):
                            all_missing = False
                            break
                    if all_missing:
                        done_reason = f"Deleted requested file(s): {', '.join(goal_tokens)}"
                        history.append(
                            {
                                "iteration": str(iteration),
                                "decision": f"done: {done_reason}",
                                "apply_result": apply_msg,
                                "verify_result": verify_msg,
                            }
                        )
                        write_plan_md(
                            repo_root=repo_root,
                            user_goal=user_goal,
                            max_iterations=args.max_iterations,
                            iteration=iteration,
                            phase="done",
                            status=f"planner declared completion: {done_reason}",
                            repo_status=git_status_short(repo_root),
                            latest_summary=done_reason,
                            history=history,
                        )
                        print(f"[planner] done: {done_reason}", file=sys.stderr)
                        return 0

    write_plan_md(
        repo_root=repo_root,
        user_goal=user_goal,
        max_iterations=args.max_iterations,
        iteration=args.max_iterations,
        phase="stopped",
        status="maximum iterations reached without planner declaring completion",
        repo_status=git_status_short(repo_root),
        latest_summary="stopped at max_iterations",
        history=history,
    )
    print("[planner] max iterations reached without successful completion.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
