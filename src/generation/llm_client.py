"""Gemini LLM client for structured tabular synthesis using Gemini Flash."""
import os
import json
import time
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
        model_name: str = "gemini-3.8-flash",
        temperature: float = 0.7,
        mock_mode: bool = False,
        strict: bool = False,
        pause_seconds: float = 1.0,
        call_cap: Optional[int] = None
    ):
        raw_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.api_key = raw_key.strip() if raw_key else None
        self.model_name = model_name
        self.temperature = temperature
        self.mock_mode = mock_mode
        self.strict = strict
        self.pause_seconds = pause_seconds
        self.call_cap = call_cap
        self.call_count = 0
        self.last_generator = "none"
        self.stats = {
            "live_calls": 0,
            "fallback_calls": 0,
            "rate_limits_429": 0,
            "errors": 0
        }

        # Treat any non-empty key as valid
        if not self.api_key:
            if self.strict:
                raise RuntimeError("Strict mode enabled: Gemini API key is missing or empty.")
            self.mock_mode = True

    def save_summary(self, filepath: Optional[Path] = None) -> None:
        """Writes execution statistics and generator provenance to JSON."""
        target_path = Path(filepath) if filepath else Path("logs/live_validation/generation_summary.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_generator": self.last_generator,
            "model_name": self.model_name,
            "mock_mode": self.mock_mode,
            "strict": self.strict,
            "call_count": self.call_count,
            "stats": self.stats
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def generate(self, prompt: str) -> str:
        """Sends a prompt to Gemini Flash requesting structured JSON output."""
        if self.mock_mode:
            if self.strict:
                raise RuntimeError("Strict mode enabled: Client is configured in mock mode.")
            self.last_generator = "mock-fallback"
            self.stats["fallback_calls"] += 1
            return self._generate_mock_response(prompt)

        if self.call_cap is not None and self.call_count >= self.call_cap:
            if self.strict:
                raise RuntimeError(f"Strict mode: API call cap ({self.call_cap}) reached.")
            self.last_generator = "mock-fallback"
            self.stats["fallback_calls"] += 1
            return self._generate_mock_response(prompt)

        # Circuit breaker: if currently in cooldown, immediately use schema generation
        cooldown = getattr(self, "_cooldown_until", 0)
        if time.time() < cooldown:
            if self.strict:
                remaining = int(cooldown - time.time())
                raise RuntimeError(f"Strict mode: Client in cooldown ({remaining}s remaining).")
            self.last_generator = "mock-fallback"
            self.stats["fallback_calls"] += 1
            return self._generate_mock_response(prompt)

        if self.pause_seconds > 0 and self.call_count > 0:
            time.sleep(self.pause_seconds)

        model = self.model_name
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": self.temperature
            }
        }

        last_error = "unknown"
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and parts[0].get("text"):
                            self.call_count += 1
                            self.stats["live_calls"] += 1
                            self.last_generator = model
                            return parts[0].get("text", "")
                    last_error = "HTTP 200 received but candidate text was empty"
                elif resp.status_code == 429:
                    self.stats["rate_limits_429"] += 1
                    last_error = f"HTTP 429 rate limit on model {model}"
                    self._cooldown_until = time.time() + 60.0
                elif resp.status_code in (400, 401, 403):
                    self.stats["errors"] += 1
                    status_desc = {400: "Bad Request", 401: "Unauthorized", 403: "Forbidden"}.get(resp.status_code, "Auth/Client Error")
                    last_error = f"HTTP {resp.status_code} ({status_desc}) on model {model}"
                    self._cooldown_until = time.time() + 3600.0
                elif resp.status_code in (404, 503):
                    self.stats["errors"] += 1
                    last_error = f"HTTP {resp.status_code} on model {model}"
                else:
                    self.stats["errors"] += 1
                    last_error = f"HTTP {resp.status_code} on model {model}"
                    self._cooldown_until = time.time() + 3600.0
        except Exception as e:
            self.stats["errors"] += 1
            last_error = f"{type(e).__name__} during request to model {model}"
            self._cooldown_until = time.time() + 300.0

        if self.strict:
            raise RuntimeError(f"Strict mode: Gemini API call failed on model '{model}': {last_error}")

        self.last_generator = "mock-fallback"
        self.stats["fallback_calls"] += 1
        print(f"[Note] Gemini API rate-limited or unavailable ({last_error}). Using schema-aware generation.")
        return self._generate_mock_response(prompt)

    def _generate_mock_response(self, prompt: str) -> str:
        """Generates realistic schema-compliant mock tabular JSON dynamically extracted from the prompt."""
        import numpy as np
        import re

        # Determine number of requested samples
        match_n = re.search(r'(?:generate|synthesize)\s+(\d+)\s+synthetic', prompt.lower())
        n_samples = int(match_n.group(1)) if match_n else 5
        n_samples = max(1, min(n_samples, 50))

        # Extract features and their ranges from Variable Description Prompt
        # Format in prompt: - feature_name (type): min=X, max=Y, mean=Z
        # or Feature 'name': min=..., max=...
        feature_bounds = {}
        for line in prompt.split("\n"):
            line_clean = line.strip()
            # Check pattern: - `ColName` (type): ... or - ColName (type): ...
            m = re.match(r'[-*]\s*[`"]?([A-Za-z0-9_\-\.]+)[`"]?\s*(?:\(([^)]*)\))?:\s*(.*)', line_clean)
            if m:
                col_name = m.group(1)
                type_str = (m.group(2) or "").lower()
                meta_str = m.group(3)
                if col_name.lower() in ("class", "target"):
                    continue
                # Extract range: [low, high] or min=X, max=Y
                m_range = re.search(r'range:\s*\[([-\d\.]+),\s*([-\d\.]+)\]', meta_str)
                m_min = re.search(r'min\s*=\s*([-\d\.]+)', meta_str)
                m_max = re.search(r'max\s*=\s*([-\d\.]+)', meta_str)
                m_mean = re.search(r'mean:\s*([-\d\.]+)', meta_str) or re.search(r'mean\s*=\s*([-\d\.]+)', meta_str)

                if m_range:
                    low = float(m_range.group(1))
                    high = float(m_range.group(2))
                else:
                    low = float(m_min.group(1)) if m_min else 0.0
                    high = float(m_max.group(1)) if m_max else 100.0

                mean_val = float(m_mean.group(1)) if m_mean else (low + high) / 2.0
                is_int = ("int" in type_str) or ("integer" in meta_str.lower())
                feature_bounds[col_name] = (low, high, mean_val, is_int)

        # Also parse exemplar values from prompt if present
        json_matches = re.findall(r'\{[^{}]+\}', prompt)
        for jm in json_matches:
            try:
                obj = json.loads(jm)
                for k, v in obj.items():
                    if k.lower() != "class" and isinstance(v, (int, float)):
                        if k not in feature_bounds:
                            feature_bounds[k] = (float(v) * 0.8, float(v) * 1.2, float(v), isinstance(v, int))
            except Exception:
                pass

        # Extract target class from prompt rules (multiclass and binary support)
        target_val = "positive"
        m_target = re.search(r'["\'](?:Class|target)["\']:\s*["\']([^"\']+)["\']', prompt, re.IGNORECASE)
        if not m_target:
            m_target = re.search(r'target class\s*["\']([^"\']+)["\']', prompt, re.IGNORECASE)
        if m_target:
            target_val = m_target.group(1)

        mock_rows = []
        for _ in range(n_samples):
            row = {}
            for col, (low, high, mean_val, is_int) in feature_bounds.items():
                if is_int:
                    row[col] = int(round(np.random.uniform(low, high)))
                else:
                    # Perturb around mean within bounds
                    val = np.random.normal(mean_val, (high - low) / 6.0)
                    row[col] = round(float(np.clip(val, low, high)), 4)
            row["Class"] = target_val
            mock_rows.append(row)

        return json.dumps(mock_rows, indent=2)
