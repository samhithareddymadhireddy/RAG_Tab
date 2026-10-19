"""Gemini LLM client for structured tabular synthesis using Gemini Flash."""
import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
import httpx
from dotenv import load_dotenv

# Automatically load environment variables from .env
env_candidates = [
    Path(".env"),
    Path("../.env"),
    Path("../../.env"),
    Path(r"c:\Users\Admin\Desktop\RAG-Tab\.env")
]
for p in env_candidates:
    if p.exists():
        load_dotenv(dotenv_path=p)
        break


class GeminiTabularClient:
    """Client for invoking Google Gemini Flash to generate structured tabular samples."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3-flash-preview",
        temperature: float = 0.7,
        mock_mode: bool = False
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.mock_mode = mock_mode

        # If no valid API key is found, automatically run in offline validation mock mode
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            self.mock_mode = True

    def generate(self, prompt: str) -> str:
        """Sends a prompt to Gemini Flash requesting structured JSON output."""
        if self.mock_mode:
            return self._generate_mock_response(prompt)

        # Use direct HTTPS REST for maximum stability across Windows and HTTP/2 proxies
        candidate_models = ["gemini-3-flash-preview", "gemini-flash-lite-latest"]
        last_error = None

        for model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": self.temperature
                }
            }
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                return parts[0].get("text", "")
                    elif resp.status_code in (404, 503, 429):
                        # Model name deprecated, busy, or rate-limited; fall through to next candidate model
                        last_error = f"HTTP {resp.status_code} on {model}"
                        continue
                    else:
                        last_error = f"HTTP {resp.status_code}: {resp.text}"
            except Exception as e:
                last_error = str(e)

        print(f"[Warning] Live Gemini API invocation failed ({last_error}). Falling back to validation mock mode.")
        return self._generate_mock_response(prompt)

    def _generate_mock_response(self, prompt: str) -> str:
        """Generates realistic schema-compliant mock tabular JSON for offline validation."""
        import numpy as np

        n_samples = 5
        if "generate 1 " in prompt.lower() or "synthesize 1 " in prompt.lower():
            n_samples = 1

        mock_rows = []
        for _ in range(n_samples):
            h = int(np.random.choice([1, 2, 5, 8, 12, 18, 25, 45]))
            l = int(np.random.choice([5, 12, 35, 75, 120, 250]))
            area = float(h * l)
            eccen = round(float(l / max(1, h)), 3)
            p_black = round(float(np.random.uniform(0.15, 0.85)), 3)
            p_and = round(float(np.random.uniform(0.30, 0.95)), 3)
            mean_tr = round(float(np.random.uniform(2.0, 15.0)), 2)
            blackpix = int(area * p_black)
            blackand = int(blackpix * 0.8)
            wb_trans = int(np.random.randint(2, 80))

            mock_rows.append({
                "Height": h,
                "Lenght": l,
                "Area": area,
                "Eccen": eccen,
                "P_black": p_black,
                "P_and": p_and,
                "Mean_tr": mean_tr,
                "Blackpix": blackpix,
                "Blackand": blackand,
                "Wb_trans": wb_trans,
                "Class": "positive"
            })

        return json.dumps(mock_rows, indent=2)
