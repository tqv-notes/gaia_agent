import os
import json
import re
import subprocess
import tempfile
import base64
import traceback
import time
import gradio as gr
import requests
import pandas as pd
from anthropic import Anthropic

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# ----------------------------------------------
#  DEOBFUSCATION
# ----------------------------------------------
COMMON_EN = {
    "the","be","to","of","and","a","in","that","have","i","it","for","not","on",
    "with","he","as","you","do","at","this","but","his","by","from","they","we",
    "say","her","she","or","an","will","my","one","all","would","there","their",
    "what","so","up","out","if","about","who","get","which","go","me","when",
    "can","like","no","just","know","take","some","could","them","see","other",
    "than","then","now","only","its","also","after","use","how","our","way",
    "even","want","because","any","these","give","most","us","is","are","was",
    "were","been","has","had","did","does","write","right","left","word",
    "opposite","understand","sentence","answer"
}
def count_en(t): return sum(1 for w in re.findall(r"[a-zA-Z]{2,}", t.lower()) if w in COMMON_EN)
def deobfuscate(t):
    if count_en(t) >= 3: return t, "none"
    r = t[::-1]
    if count_en(r) > count_en(t): return r, "reversed"
    return t, "none"

# ----------------------------------------------
#  DETERMINISTIC FILE PREPROCESSING
# ----------------------------------------------
def preprocess_file(filepath):
    if not filepath or not os.path.exists(filepath):
        return None
    ext = os.path.splitext(filepath)[1].lower()
    print(f"  [PREPROCESS] {filepath} ({ext})")

    if ext == ".py":
        try:
            code = open(filepath).read()
            result = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=120,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            output = result.stdout.strip()
            stderr = result.stderr.strip()
            ctx = f"Python file content:\n```python\n{code}\n```\n\nExecution output:\n{output}"
            if stderr: ctx += f"\nStderr: {stderr}"
            print(f"  [PREPROCESS] Python output: {output[:200]}")
            return ctx
        except Exception as e:
            try:
                code = open(filepath).read()
                return f"Python file (execution failed):\n```python\n{code}\n```\nError: {e}"
            except: return f"Python file at {filepath}, error: {e}"

    if ext in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(filepath)
            s = f"Excel file. Shape: {df.shape}\nColumns: {df.columns.tolist()}\n\nFirst 30 rows:\n{df.head(30).to_string()}\n\nData types:\n{df.dtypes.to_string()}"
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                s += f"\n\nColumn sums:\n{df[num_cols].sum().to_string()}"
            print(f"  [PREPROCESS] Excel: {df.shape}")
            return s
        except Exception as e:
            return f"Excel file at {filepath}, error: {e}"

    if ext == ".csv":
        try:
            df = pd.read_csv(filepath)
            return f"CSV. Shape: {df.shape}\nColumns: {df.columns.tolist()}\nFirst 30 rows:\n{df.head(30).to_string()}"
        except Exception as e:
            return f"CSV file at {filepath}, error: {e}"

    if ext == ".pdf":
        try:
            result = subprocess.run(
                ["python3", "-c", f"import PyPDF2;r=PyPDF2.PdfReader('{filepath}');\nfor p in r.pages: print(p.extract_text())"],
                capture_output=True, text=True, timeout=60)
            return f"PDF content:\n{result.stdout.strip()[:15000]}"
        except Exception as e:
            return f"PDF at {filepath}, error: {e}"

    if ext in (".mp3", ".wav", ".ogg", ".m4a"):
        try:
            print(f"  [PREPROCESS] Whisper transcription...")
            result = subprocess.run(
                ["python3", "-c",
                 f"import warnings;warnings.filterwarnings('ignore');"
                 f"import whisper;m=whisper.load_model('tiny');"
                 f"r=m.transcribe('{filepath}');print(r['text'])"],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            transcript = result.stdout.strip()
            if transcript:
                print(f"  [PREPROCESS] Transcript: {transcript[:200]}")
                return f"Audio transcription:\n{transcript}"
            return f"Audio file at {filepath}. Whisper produced no output. stderr: {result.stderr[:300]}"
        except subprocess.TimeoutExpired:
            return f"Audio at {filepath}. Transcription timed out."
        except Exception as e:
            return f"Audio at {filepath}. Error: {e}"

    if ext in (".txt", ".json", ".md"):
        try:
            return f"File content:\n{open(filepath).read()[:15000]}"
        except Exception as e:
            return f"Text file at {filepath}, error: {e}"

    return f"File at {filepath} (type: {ext}). Use python_execute to examine."

# ----------------------------------------------
#  ANSWER CLEANUP (regex only, NO LLM)
# ----------------------------------------------
def clean_answer(raw):
    ans = raw.strip()
    # remove markdown bold
    ans = re.sub(r'\*\*(.+?)\*\*', r'\1', ans)
    ans = ans.strip('*"\'`').strip()
    # remove common verbose prefixes
    for prefix in ["The answer is ", "The final answer is ", "Answer: ", "FINAL ANSWER: ",
                    "Based on my analysis, ", "Based on the data, ", "The result is "]:
        if ans.lower().startswith(prefix.lower()):
            ans = ans[len(prefix):].strip()
    # remove trailing period (but not decimal like 89706.00)
    if ans.endswith('.') and not re.match(r'^\d+\.\d+$', ans):
        ans = ans[:-1].strip()
    # remove emoji
    ans = re.sub(r'[\U0001F300-\U0001F9FF]', '', ans).strip()
    return ans

# ----------------------------------------------
#  SEARCH TOOLS (Tavily primary, DDG fallback)
# ----------------------------------------------
def tool_web_search(query):
    """try Tavily first, fall back to DuckDuckGo."""
    # Tavily (primary)
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            resp = requests.post("https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": query, "max_results": 5,
                      "include_answer": True, "search_depth": "basic"},
                timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                parts = []
                if data.get("answer"):
                    parts.append(f"Direct answer: {data['answer']}")
                for i, r in enumerate(data.get("results", [])[:5], 1):
                    parts.append(f"[{i}] {r.get('title','')}\n    URL: {r.get('url','')}\n    {r.get('content','')[:300]}")
                result = "\n\n".join(parts)
                if result.strip():
                    return result
        except Exception as e:
            print(f"    Tavily error: {e}")

    # DuckDuckGo (fallback)
    try:
        from duckduckgo_search import DDGS
        with DDGS() as d:
            results = list(d.text(query, max_results=7))
        if not results:
            return "No results found."
        return "\n\n".join(f"[{i+1}] {r['title']}\n    URL: {r['href']}\n    {r['body']}" for i, r in enumerate(results))
    except Exception as e:
        return f"Search error: {e}"


def tool_wikipedia(query):
    try:
        import wikipedia
        from bs4 import BeautifulSoup
        try:
            page = wikipedia.page(query, auto_suggest=True)
        except wikipedia.DisambiguationError as e:
            try: page = wikipedia.page(e.options[0])
            except: return f"Disambiguation: {e.options[:5]}"
        except wikipedia.PageError:
            r = wikipedia.search(query, results=3)
            if r:
                try: page = wikipedia.page(r[0])
                except: return f"Found: {r}"
            else: return "Not found."
        soup = BeautifulSoup(page.html(), "html.parser")
        c = soup.find("div", class_="mw-parser-output")
        if not c: return page.content[:15000]
        for t in c.find_all(["style", "script", "sup"]): t.decompose()
        for t in c.find_all(class_=["navbox", "reference"]): t.decompose()
        return c.get_text(separator="\n", strip=True)[:20000]
    except Exception as e:
        return f"Wikipedia error: {e}"


def tool_visit(url):
    try:
        import ssl, urllib.request
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "nav", "footer"]): t.decompose()
        return soup.get_text(separator="\n", strip=True)[:20000]
    except Exception as e:
        # fallback with requests
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
            r.raise_for_status()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "html.parser")
            for t in soup(["script", "style", "nav", "footer"]): t.decompose()
            return soup.get_text(separator="\n", strip=True)[:20000]
        except Exception as e2:
            return f"Error: {e2}"


def tool_yt_transcript(url):
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        m = re.search(r"(?:v=|youtu\.be/)([^&?]+)", url)
        if not m: return f"No ID in {url}"
        ytt = YouTubeTranscriptApi()
        t = ytt.fetch(m.group(1))
        return "\n".join(s.text for s in t)[:20000]
    except Exception as e:
        # try noembed for at least the title
        try:
            import urllib.request, json
            noembed = f"https://noembed.com/embed?url={url}"
            data = json.loads(urllib.request.urlopen(noembed, timeout=10).read())
            title = data.get("title", "")
            author = data.get("author_name", "")
            info = f"Video title: {title}\nAuthor: {author}"
        except:
            info = ""

        return (f"Transcript error: {e}\n\n{info}\n\n"
                f"FALLBACK: Use python_execute to download and transcribe:\n"
                f"import yt_dlp, whisper, os, tempfile, ssl\n"
                f"ssl._create_default_https_context = ssl._create_unverified_context\n"
                f"tmp=tempfile.mkdtemp()\n"
                f"o={{'format':'bestaudio','outtmpl':os.path.join(tmp,'a.%(ext)s'),"
                f"'postprocessors':[{{'key':'FFmpegExtractAudio','preferredcodec':'wav'}}],'quiet':True}}\n"
                f"with yt_dlp.YoutubeDL(o) as y: y.extract_info('{url}',download=True)\n"
                f"wav=[os.path.join(tmp,f) for f in os.listdir(tmp) if f.endswith('.wav')][0]\n"
                f"print(whisper.load_model('tiny').transcribe(wav)['text'])")


def tool_python(code):
    try:
        r = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=180,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        parts = []
        if r.stdout: parts.append(r.stdout.strip())
        if r.stderr: parts.append(f"[STDERR]: {r.stderr.strip()}")
        out = "\n".join(parts)
        return out[:15000] if out else "(No output)"
    except subprocess.TimeoutExpired: return "Timed out."
    except Exception as e: return f"Error: {e}"


# ----------------------------------------------
#  TOOLS
# ----------------------------------------------
TOOLS = [
    {"name": "web_search", "description": "Search the web (Tavily + DuckDuckGo). Use for facts, statistics, current events. Try different queries if first fails.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "wikipedia_search", "description": "Get full Wikipedia page content. Try FIRST for factual questions about people, events, places, competitions, species.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "visit_webpage", "description": "Fetch full text of a webpage URL. Works with SSL issues. Use after web_search to read full articles.",
     "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    {"name": "youtube_transcript", "description": "Get YouTube video transcript. ALWAYS try for YouTube URLs.",
     "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    {"name": "python_execute", "description": "Execute Python code. Use for calculations, data analysis, string operations, audio transcription. Use print().",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
]

DISPATCH = {
    "web_search": lambda a: tool_web_search(a["query"]),
    "wikipedia_search": lambda a: tool_wikipedia(a["query"]),
    "visit_webpage": lambda a: tool_visit(a["url"]),
    "youtube_transcript": lambda a: tool_yt_transcript(a["url"]),
    "python_execute": lambda a: tool_python(a["code"]),
}

# ----------------------------------------------
#  SYSTEM PROMPT
# ----------------------------------------------
SYSTEM_PROMPT = """You are an expert AI assistant. Answer questions precisely and concisely.

## ANSWER FORMAT (graded by EXACT STRING MATCH):
End with: FINAL ANSWER: [answer]

Rules:
- SHORT answers only: a number, a name, a place, or a brief comma-separated list.
- No markdown, no bold, no quotes, no explanations after FINAL ANSWER:
- "How many..." => just the number
- "Who..." => just the name
- "Where..." => just the place name
- Dollar amounts => number with 2 decimals
- Lists => comma-separated, alphabetical unless specified otherwise
- NEVER say "Unable to determine". Always give your best answer even if uncertain.
- NEVER ask for files to be re-uploaded. Work with what you have.

## STRATEGY:
1. If file content is provided in the context below, USE IT directly — do NOT ask for files.
2. Start with wikipedia_search for factual/historical questions.
3. Use web_search for anything else. Try 3+ DIFFERENT queries if needed.
4. Use visit_webpage to read full articles when search snippets aren't enough.
5. For YouTube videos: use youtube_transcript. If it fails, try python_execute with yt_dlp+whisper.
6. Use python_execute for calculations, data processing.
7. NEVER repeat the exact same search query.
8. For multi-step questions, break them down and solve each step.
9. If all tools fail, make your BEST GUESS based on available information.
"""

# ----------------------------------------------
#  AGENT
# ----------------------------------------------
class GaiaAgent:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key: raise ValueError("ANTHROPIC_API_KEY required.")
        self.client = Anthropic(api_key=api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.max_steps = 20
        print(f"GaiaAgent: {self.model}")

    def _api_call(self, **kw):
        for a in range(5):
            try:
                return self.client.messages.create(**kw)
            except Exception as e:
                err = str(e)
                if "rate_limit" in err or "429" in err:
                    w = 25 * (a + 1); print(f"    Rate limited, {w}s..."); time.sleep(w)
                elif "overloaded" in err or "529" in err:
                    time.sleep(30 * (a + 1))
                elif "credit" in err.lower() or "billing" in err.lower():
                    print(f"    OUT OF CREDITS: {err}")
                    raise
                else:
                    raise
        raise Exception("Max retries exceeded.")

    def _extract(self, text):
        for p in [r"FINAL ANSWER:\s*(.+?)$", r"FINAL ANSWER:\s*(.+)"]:
            m = re.search(p, text, re.MULTILINE | re.IGNORECASE)
            if m: return m.group(1).strip()
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else text.strip()

    def run(self, question, file_path=None, file_context=None):
        time.sleep(3)

        decoded, method = deobfuscate(question)
        if method != "none":
            print(f"  Deobfuscated ({method}): {decoded[:80]}...")

        user_content = []

        # attach image if applicable
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                try:
                    with open(file_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    mt = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                          ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/png")
                    user_content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}})
                    print(f"  Image attached")
                except Exception as e:
                    print(f"  Image error: {e}")

        # build prompt
        prompt = decoded
        if file_context:
            prompt = f"FILE CONTEXT (already preprocessed — use this data directly, do NOT ask for files):\n{file_context}\n\n---\nQuestion: {decoded}"

        user_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": user_content}]

        for step in range(self.max_steps):
            print(f"  Step {step+1}/{self.max_steps}...")
            try:
                resp = self._api_call(model=self.model, max_tokens=4096, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages)
            except Exception as e:
                print(f"  API error: {e}")
                return f"Error: {e}"

            if resp.stop_reason == "end_turn":
                text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                return self._extract(text)

            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for b in resp.content:
                    if b.type == "tool_use":
                        print(f"    Tool: {b.name}({json.dumps(b.input)[:100]})")
                        r = DISPATCH.get(b.name, lambda x: "Unknown")(b.input)
                        print(f"    Result: {str(r)[:200]}...")
                        results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(r)})
                messages.append({"role": "user", "content": results})
            else:
                text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                if text: return self._extract(text)
                break

        return "Unable to determine answer."


# ----------------------------------------------
#  FILE DOWNLOAD
# ----------------------------------------------
def download_file(task_id, file_name=None):
    """download file from HuggingFace dataset (primary) or GAIA API (fallback)."""
    d = os.path.join(tempfile.gettempdir(), "gaia_files")
    os.makedirs(d, exist_ok=True)

    # primary: HuggingFace dataset
    if file_name:
        hf_url = f"https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/{file_name}"
        headers = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        for attempt in range(3):
            try:
                print(f"  [DL] HF dataset: {hf_url} (attempt {attempt+1})")
                r = requests.get(hf_url, timeout=30, allow_redirects=True, headers=headers)
                print(f"  [DL] {r.status_code} Size={len(r.content)}b")
                if r.status_code == 200 and len(r.content) > 0:
                    p = os.path.join(d, file_name)
                    with open(p, "wb") as f:
                        f.write(r.content)
                    print(f"  [DL] Saved: {p} ({len(r.content)/1024:.1f}KB)")
                    return p
                if attempt < 2: time.sleep(2)
            except Exception as e:
                print(f"  [DL] HF error: {e}")
                if attempt < 2: time.sleep(2)

    # fallback: GAIA scoring API
    url = f"{DEFAULT_API_URL}/files/{task_id}"
    try:
        print(f"  [DL] API fallback: {url}")
        r = requests.get(url, timeout=30)
        print(f"  [DL] {r.status_code} Size={len(r.content)}b")
        if r.status_code == 200 and len(r.content) > 0:
            fn = file_name or f"task_{task_id}.bin"
            p = os.path.join(d, fn)
            with open(p, "wb") as f:
                f.write(r.content)
            print(f"  [DL] Saved: {p} ({len(r.content)/1024:.1f}KB)")
            return p
    except Exception as e:
        print(f"  [DL] API error: {e}")

    print(f"  [DL] All download attempts failed")
    return None


# ----------------------------------------------
#  SUBMISSION
# ----------------------------------------------
def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")
    if not profile: return "Please Login.", None
    username = profile.username
    try:
        agent = GaiaAgent()
    except Exception as e:
        return f"Error: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"

    try:
        r = requests.get(f"{DEFAULT_API_URL}/questions", timeout=15)
        r.raise_for_status()
        questions = r.json()
        if not questions: return "No questions.", None
        print(f"Fetched {len(questions)} questions.")
        for q in questions:
            print(f"  {q.get('task_id', '')} file={q.get('file_name', 'None')}")
    except Exception as e:
        return f"Error: {e}", None

    results_log, answers = [], []
    for i, item in enumerate(questions):
        task_id = item.get("task_id")
        q = item.get("question")
        file_name = item.get("file_name")
        if not task_id or q is None: continue
        print(f"\n{'='*60}\n[{i+1}/{len(questions)}] {task_id} file={file_name}\nQ: {q[:100]}...")

        # download file
        file_path = download_file(task_id, file_name) if file_name else None
        if not file_path and file_name:
            # file_name exists but download failed - still try without file
            print(f"  File download failed for {file_name}")

        # preprocess file
        file_context = preprocess_file(file_path) if file_path else None
        if file_context: print(f"  Preprocessed: {len(file_context)} chars")

        # run agent
        try:
            raw = agent.run(q, file_path=file_path, file_context=file_context)
            print(f"  Raw: {raw[:200]}")
            cleaned = clean_answer(raw)
            print(f"  Cleaned: {cleaned}")
            answers.append({"task_id": task_id, "submitted_answer": cleaned})
            results_log.append({"Task ID": task_id, "Question": q[:100], "Submitted Answer": cleaned})
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results_log.append({"Task ID": task_id, "Question": q[:100], "Submitted Answer": f"ERROR: {e}"})

    if not answers: return "No answers.", pd.DataFrame(results_log)

    data = {"username": username.strip(), "agent_code": agent_code, "answers": answers}
    try:
        r = requests.post(f"{DEFAULT_API_URL}/submit", json=data, timeout=120)
        r.raise_for_status()
        d = r.json()
        return (f"Submission Successful!\nUser: {d.get('username')}\n"
                f"Score: {d.get('score', 'N/A')}% ({d.get('correct_count', '?')}/{d.get('total_attempted', '?')} correct)\n"
                f"Message: {d.get('message', '')}"), pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)


with gr.Blocks() as demo:
    gr.Markdown("# GAIA Benchmark Agent")
    gr.Markdown("Set `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` in Space secrets. Log in and click Run.")
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Status", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Results", wrap=True)
    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

if __name__ == "__main__":
    print(f"{'✅' if os.getenv('ANTHROPIC_API_KEY') else '⚠️'} ANTHROPIC_API_KEY")
    print(f"{'✅' if os.getenv('TAVILY_API_KEY') else '⚠️'} TAVILY_API_KEY")
    demo.launch(debug=True, share=False)