"""Narrate the demo video with ElevenLabs, then rebuild it.

Usage: PYTHONPATH=. uv run python scripts/tts_elevenlabs.py [voice name, default "George"]
Needs ELEVENLABS_API_KEY in .env. Writes out/video/n1..n6.aiff and runs out/video/build.sh.
"""
import subprocess
import sys
from pathlib import Path

import requests
from dotenv import dotenv_values

KEY = dotenv_values(".env").get("ELEVENLABS_API_KEY")
if not KEY:
    raise SystemExit("ELEVENLABS_API_KEY missing from .env")
H = {"xi-api-key": KEY}
VID = Path("out/video")
want = sys.argv[1] if len(sys.argv) > 1 else "George"
TEMPO = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0  # speed-up so the video stays under 60 s
PREMADE = {"george": "JBFqnCBsd6RMkjVDRZzb", "rachel": "21m00Tcm4TlvDq8ikWAM", "adam": "pNInz6obpgDQGcFmaJgB"}
voices = requests.get("https://api.elevenlabs.io/v1/voices", headers=H, timeout=30).json().get("voices", [])
voice = next((v for v in voices if v["name"].lower().startswith(want.lower()) or v["voice_id"] == want), None)
if not voice and len(want) == 20:  # a voice id from the shared library (e.g. Neel, Indian accent)
    voice = {"name": want, "voice_id": want}
if not voice:  # key without voices_read permission: use a premade voice by id
    voice = {"name": want, "voice_id": PREMADE.get(want.lower(), PREMADE["george"])}
print("voice:", voice["name"], voice["voice_id"])
lines = [l.strip() for l in (VID / "narration.txt").read_text().splitlines() if l.strip()]
for i, text in enumerate(lines, 1):
    r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice['voice_id']}?output_format=mp3_44100_128",
                      headers={**H, "Content-Type": "application/json"}, timeout=120,
                      json={"text": text, "model_id": "eleven_multilingual_v2",
                            "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.25,
                                               "use_speaker_boost": True}})
    if r.status_code != 200:
        raise SystemExit(f"line {i}: HTTP {r.status_code} {r.text[:200]}")
    mp3 = VID / f"n{i}.mp3"
    mp3.write_bytes(r.content)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-filter:a", f"atempo={TEMPO}",
                    str(VID / f"n{i}.aiff")], check=True)
    print(f"n{i}: {len(text)} chars")
subprocess.run(["bash", "build.sh"], cwd=VID, check=True)
