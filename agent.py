"""
AMD Hackathon - Track 2: Video Captioning Agent

Reads /input/tasks.json, produces /output/results.json.

Design priorities, in order:
  1. ALWAYS write a valid results.json with every requested style present.
     A missing style scores zero; a mediocre caption does not.
  2. Stay inside the 10-minute wall clock (clips are processed in parallel,
     and a global deadline forces a write-out no matter what).
  3. Caption quality.
"""

import base64
import concurrent.futures as futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

# ---------------------------------------------------------------- config

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL}:generateContent"
)

ALL_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

N_FRAMES = 8            # frames sampled per clip
FRAME_WIDTH = 512       # downscale: keeps upload fast, plenty for the judge
MAX_WORKERS = 6         # clips processed concurrently
DOWNLOAD_TIMEOUT = 120
FFMPEG_TIMEOUT = 90
API_TIMEOUT = 90
GLOBAL_DEADLINE = 8.5 * 60   # seconds; hard stop well inside the 10-min limit

START = time.time()


def remaining():
    return GLOBAL_DEADLINE - (time.time() - START)


def log(msg):
    print(f"[{time.time() - START:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- prompting

STYLE_RULES = """\
formal            - Professional, objective, factual. No jokes, no opinion, no flourish.
sarcastic         - Dry, ironic, lightly mocking. Deadpan understatement. Never cruel.
humorous_tech     - Funny, using a programming/technology metaphor that actually fits
                    what is on screen (e.g. memory leaks, merge conflicts, buffering,
                    infinite loops). The tech reference must map to the real content.
humorous_non_tech - Funny, everyday observational humour. ZERO technical or computing
                    words. If you mention code, servers, or software here, it is wrong.
"""

PROMPT = f"""You are captioning a short video clip. You are shown {N_FRAMES} frames sampled evenly across it, in chronological order.

STEP 1 - Observe. Identify concretely: the main subject(s), the setting, any motion or
action progressing across the frames, and 2-3 distinctive visual details (colours, weather,
objects, time of day, camera movement).

STEP 2 - Write four captions of the SAME clip, one per style. Every caption must be
grounded in what is actually visible - a caption that could describe any random video
scores zero on accuracy. Each caption is 1-2 sentences, English.

STYLES:
{STYLE_RULES}
The four captions must describe the SAME observed content; only the tone changes.
Do not invent people, dialogue, brands, or events that are not visible.

Return ONLY a JSON object, no markdown fences:
{{"description": "<one factual sentence of what happens>",
  "captions": {{"formal": "...", "sarcastic": "...", "humorous_tech": "...", "humorous_non_tech": "..."}}}}
"""


# ---------------------------------------------------------------- video -> frames

def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return path


def probe_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        d = float(out)
        return d if d > 0 else None
    except Exception:
        return None


def extract_frames(video_path, workdir):
    """Sample N_FRAMES evenly across the clip, downscaled, as JPEGs."""
    duration = probe_duration(video_path)
    # fps that yields ~N_FRAMES over the whole clip
    fps = (N_FRAMES / duration) if duration else 0.2
    pattern = os.path.join(workdir, "f_%02d.jpg")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video_path,
         "-vf", f"fps={fps:.6f},scale={FRAME_WIDTH}:-2",
         "-frames:v", str(N_FRAMES), "-q:v", "4", pattern],
        check=True, capture_output=True, timeout=FFMPEG_TIMEOUT,
    )
    frames = sorted(
        os.path.join(workdir, f) for f in os.listdir(workdir) if f.endswith(".jpg")
    )
    if not frames:
        raise RuntimeError("ffmpeg produced no frames")
    return frames[:N_FRAMES]


# ---------------------------------------------------------------- model call

def call_gemini(frame_paths):
    parts = []
    for p in frame_paths:
        with open(p, "rb") as f:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(f.read()).decode(),
                }
            })
    parts.append({"text": PROMPT})

    body = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.7,
            "responseMimeType": "application/json",
            "maxOutputTokens": 800,
        },
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": API_KEY},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=API_TIMEOUT) as r:
        payload = json.loads(r.read().decode())

    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    return parse_captions(text)


def parse_captions(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)   # last-ditch: grab the JSON blob
        if not m:
            raise
        obj = json.loads(m.group(0))
    caps = obj.get("captions", obj)
    return {s: str(caps[s]).strip() for s in ALL_STYLES if caps.get(s)}


# ---------------------------------------------------------------- fallbacks

def fallback(styles, hint=""):
    """Never emit a missing style. Generic, but non-zero."""
    subject = hint or "the scene"
    base = {
        "formal": f"The clip presents {subject}, captured in a continuous shot.",
        "sarcastic": f"Ah yes, {subject}. Riveting stuff, truly.",
        "humorous_tech": f"POV: {subject} renders at a stable 60fps and still won't fix my build.",
        "humorous_non_tech": f"Somewhere out there, {subject} is having a better day than me.",
    }
    return {s: base.get(s, base["formal"]) for s in styles}


# ---------------------------------------------------------------- per-task

def process(task):
    tid = task.get("task_id")
    styles = task.get("styles") or ALL_STYLES
    url = task.get("video_url")
    log(f"{tid}: start")

    try:
        if remaining() < 45:
            raise TimeoutError("global deadline reached before start")

        with tempfile.TemporaryDirectory() as wd:
            video = download(url, os.path.join(wd, "clip.mp4"))
            frames = extract_frames(video, wd)
            log(f"{tid}: {len(frames)} frames -> model")

            caps = {}
            for attempt in (1, 2):
                try:
                    caps = call_gemini(frames)
                    if all(s in caps for s in styles):
                        break
                except Exception as e:
                    log(f"{tid}: attempt {attempt} failed: {e}")
                    if attempt == 2 or remaining() < 40:
                        break
                    time.sleep(2)

        if not all(s in caps for s in styles):
            missing = [s for s in styles if s not in caps]
            log(f"{tid}: filling missing styles {missing}")
            caps.update({s: v for s, v in fallback(missing).items()})

        log(f"{tid}: done")
        return {"task_id": tid, "captions": {s: caps[s] for s in styles}}

    except Exception as e:
        log(f"{tid}: FAILED ({e}) - emitting fallback")
        return {"task_id": tid, "captions": fallback(styles)}


# ---------------------------------------------------------------- main

def main():
    if not API_KEY:
        log("WARNING: GEMINI_API_KEY is empty - every clip will fall back.")

    with open(INPUT_PATH) as f:
        tasks = json.load(f)
    log(f"loaded {len(tasks)} task(s)")

    results = [None] * len(tasks)
    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fut_map = {pool.submit(process, t): i for i, t in enumerate(tasks)}
        for fut in futures.as_completed(fut_map, timeout=max(remaining(), 10)):
            i = fut_map[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                log(f"task {i} blew up: {e}")

    # backfill anything that never returned
    for i, t in enumerate(tasks):
        if results[i] is None:
            results[i] = {
                "task_id": t.get("task_id"),
                "captions": fallback(t.get("styles") or ALL_STYLES),
            }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"wrote {OUTPUT_PATH} ({len(results)} results)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Absolute last resort: still write something, still exit 0.
        log(f"FATAL: {e}")
        try:
            with open(INPUT_PATH) as f:
                tasks = json.load(f)
        except Exception:
            tasks = []
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(
                [{"task_id": t.get("task_id"),
                  "captions": fallback(t.get("styles") or ALL_STYLES)} for t in tasks],
                f, indent=2,
            )
        sys.exit(0)
