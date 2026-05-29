import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import nvidia_nim_worker

DEFAULT_PLANNER_MODEL = "deepseek-ai/deepseek-v4-flash"
DEFAULT_MAX_ITERATIONS = 6

PLANNER_SYSTEM_PROMPT = """You are a planning and validation agent for autonomous coding.
You collaborate with a separate coding worker that returns unified diff patches only.
Your job each iteration:
1) Read the original user goal and current repository state summary.
2) Decide the next worker task.
3) If the goal is completed, declare completion.

Return strictly valid JSON only:
{
  "action": "run_worker" | "done",
  "worker_task": "string",
  "done_reason": "string"
}

Rules:
- If action is "run_worker", worker_task must be specific and implementation-focused.
- Keep worker_task short and concrete, referencing exact files when known.
- If validation or apply failed, worker_task must include a fix strategy.
- If action is "done", done_reason must briefly explain why the goal is satisfied.
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
    return out.strip()


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


def build_planner_messages(
    user_goal: str,
    history: list[dict[str, str]],
    repo_status: str,
) -> list[dict[str, str]]:
    lines: list[str] = [f"User goal:\n{user_goal}"]
    if repo_status:
        lines.append(f"Current git status (short):\n{repo_status}")
    if history:
        entries: list[str] = []
        for item in history:
            entries.append(
                "\n".join(
                    [
                        f"Iteration: {item['iteration']}",
                        f"Planner task: {item['planner_task']}",
                        f"Worker result: {item['worker_result']}",
                        f"Apply result: {item['apply_result']}",
                        f"Verify result: {item['verify_result']}",
                    ]
                )
            )
        lines.append("Iteration history:\n" + "\n\n".join(entries))
    else:
        lines.append("Iteration history:\n(none)")
    lines.append("Decide the next action now.")
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(lines)},
    ]


def planner_step(client, args, messages: list[dict[str, str]]) -> dict[str, str]:
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
    return parsed


def run_worker(worker_script: str, worker_task: str, repo_root: Path) -> tuple[int, str, str]:
    cmd = [sys.executable, worker_script, "--task", worker_task]
    return run_cmd(cmd, repo_root)


def apply_patch(repo_root: Path, patch_text: str) -> tuple[bool, str]:
    patch = nvidia_nim_worker.clean_patch_output(patch_text).strip()
    if "diff --git " not in patch:
        return False, "Worker output did not include a git patch."

    check = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=repo_root,
        input=patch,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        return False, f"git apply --check failed:\n{check.stderr.strip() or check.stdout.strip()}"

    apply_run = subprocess.run(
        ["git", "apply", "-"],
        cwd=repo_root,
        input=patch,
        capture_output=True,
        text=True,
    )
    if apply_run.returncode != 0:
        return False, f"git apply failed:\n{apply_run.stderr.strip() or apply_run.stdout.strip()}"
    return True, "Patch applied."


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


def main() -> int:
    nvidia_nim_worker.load_env_files()
    parser = argparse.ArgumentParser(
        description="Autonomous planner that uses DeepSeek + local_llm_worker for iterative coding.",
    )
    parser.add_argument("--task", help="Top-level user goal. If omitted, read from stdin.")
    parser.add_argument("--repo-root", help="Repository root. Defaults to git root or cwd.")
    parser.add_argument("--worker-script", default="local_llm_worker.py", help="Worker script path.")
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
    worker_script = str((repo_root / args.worker_script).resolve())

    if not Path(worker_script).exists():
        raise RuntimeError(f"Worker script not found: {worker_script}")

    client = nvidia_nim_worker.build_client(args)
    history: list[dict[str, str]] = []

    print(f"[planner] repo_root={repo_root}", file=sys.stderr)
    print(f"[planner] worker_script={worker_script}", file=sys.stderr)
    print(f"[planner] max_iterations={args.max_iterations}", file=sys.stderr)

    for i in range(1, args.max_iterations + 1):
        repo_status = git_status_short(repo_root)
        messages = build_planner_messages(user_goal=user_goal, history=history, repo_status=repo_status)
        decision = planner_step(client, args, messages)

        if decision["action"] == "done":
            print(f"[planner] done at iteration {i}: {decision.get('done_reason', '')}", file=sys.stderr)
            return 0

        planner_task = decision["worker_task"].strip()
        print(f"[planner] iteration {i} task: {planner_task}", file=sys.stderr)

        code, worker_out, worker_err = run_worker(worker_script, planner_task, repo_root)
        worker_err = (worker_err or "").strip()
        worker_out = (worker_out or "").strip()
        worker_result = f"exit={code}; stderr={worker_err[:2000]}"

        if code != 0:
            history.append(
                {
                    "iteration": str(i),
                    "planner_task": planner_task,
                    "worker_result": worker_result,
                    "apply_result": "not attempted",
                    "verify_result": "not attempted",
                }
            )
            continue

        applied, apply_msg = apply_patch(repo_root, worker_out)
        if not applied:
            history.append(
                {
                    "iteration": str(i),
                    "planner_task": planner_task,
                    "worker_result": "exit=0",
                    "apply_result": apply_msg[:3000],
                    "verify_result": "not attempted",
                }
            )
            continue

        verify_ok, verify_msg = run_verify_commands(repo_root, args.verify_cmd)
        history.append(
            {
                "iteration": str(i),
                "planner_task": planner_task,
                "worker_result": "exit=0",
                "apply_result": apply_msg,
                "verify_result": verify_msg[:3000],
            }
        )

        if verify_ok:
            print(f"[planner] success at iteration {i}", file=sys.stderr)
            return 0

    print("[planner] max iterations reached without successful completion.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
