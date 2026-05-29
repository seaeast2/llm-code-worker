import nvidia_nim_worker


LOCAL_BASE_URL = "http://192.168.0.19:8080"
LOCAL_MODEL = "qwen2.5-coder-7b-instruct-q6_k"


def main() -> int:
    nvidia_nim_worker.DEFAULT_BASE_URL = LOCAL_BASE_URL
    nvidia_nim_worker.DEFAULT_BASE_URL_ENV = "LOCAL_LLM_API_BASE"
    nvidia_nim_worker.DEFAULT_API_KEY_ENV = "LOCAL_LLM_API_KEY"
    nvidia_nim_worker.DEFAULT_ALLOW_MISSING_API_KEY = True
    nvidia_nim_worker.DEFAULT_MODEL = LOCAL_MODEL
    nvidia_nim_worker.DEFAULT_MAX_TOKENS = 4096
    nvidia_nim_worker.DEFAULT_TEMPERATURE = 0.1
    nvidia_nim_worker.DEFAULT_TOP_P = 0.9
    nvidia_nim_worker.DEFAULT_REASONING_BUDGET = 0
    return nvidia_nim_worker.main()


if __name__ == "__main__":
    raise SystemExit(main())
