import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
venv_site = repo_root / ".venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=repo_root / ".env")

key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
is_present = bool(key and len(key.strip()) > 0)

print(f"Key present: {is_present}", flush=True)

from src.generation.llm_client import GeminiTabularClient

client = GeminiTabularClient()
print(f"mock_mode: {client.mock_mode}", flush=True)
