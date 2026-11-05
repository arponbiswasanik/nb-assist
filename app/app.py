"""NB Assist chat UI.

Single-file Gradio app. The chat is the primary experience; the decision
trace lives in a slide-in drawer that stays hidden until it is opened.

Usage:
  - Start Postgres first: docker compose up -d
  - Make sure Ollama is running with gemma3:4b and bge-m3 pulled
  - python app/app.py
  - Open the printed local URL (usually http://127.0.0.1:7860)

Design system: atmospheric navy/purple gradients, monochrome controls,
and translucent glass surfaces. UI styling lives in the CSS token block;
the theme object keeps Gradio's own chrome consistent with it. The chat
avatars are generated as SVGs into app/assets/ on first run.
"""

from __future__ import annotations

import csv
import html
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from nb_assist.llm import OllamaClient, LLMUnavailable
from nb_assist.retrieve import HybridRetriever
from nb_assist.pipeline import handle_query
from nb_assist.storage import Storage


# ---------- shared state built once at startup --------------------------

CONFIG = yaml.safe_load((ROOT / "config.yaml").open(encoding="utf-8"))
CLIENT = OllamaClient(CONFIG)
STORAGE = None
RETRIEVER = None
FAQ_LOOKUP = {}
ANSWER_BANK = {}
IDK_BANK = {}


def _load_csv(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_faq_lookup(rows):
    return {int(r["id"]): {"question": r["question"], "answer": r["answer"]}
            for r in rows}


def _load_answer_bank(rows):
    out = {}
    for r in rows:
        out[int(r["faq_id"])] = {
            "en": r["answer_en"],
            "bn": r["answer_bn"],
            "banglish": r["answer_banglish"],
        }
    return out


def _load_idk(rows):
    return {r["language"]: r["text"] for r in rows}


def _connect_storage():
    global STORAGE
    if not CONFIG.get("database"):
        return
    try:
        s = Storage.from_config(CONFIG).connect()
        with s._conn.cursor() as cur:
            cur.execute("SELECT 1")
        STORAGE = s
        print("[app] storage: Postgres connected (loading embeddings from DB)")
    except Exception as e:
        print(f"[app] storage: Postgres not available ({e}); "
              "falling back to CSV-built embeddings")
        STORAGE = None


def _build_retriever():
    global RETRIEVER
    faq_rows = _load_csv(CONFIG["paths"]["faq"])
    paraphrase_rows = _load_csv(CONFIG["paths"]["paraphrase_index"])

    print("[app] building retriever index...")
    t0 = time.perf_counter()
    if STORAGE is not None:
        db_faq = STORAGE.load_faq()
        db_paraphrases = STORAGE.load_paraphrases()
        if db_faq and db_faq[0].get("embedding"):
            RETRIEVER = HybridRetriever(CONFIG, CLIENT).build_index_from_storage(
                db_faq, db_paraphrases)
        else:
            print("[app] DB has no embeddings; building from CSV via Ollama")
            RETRIEVER = HybridRetriever(CONFIG, CLIENT).build_index(
                faq_rows, paraphrase_rows)
    else:
        RETRIEVER = HybridRetriever(CONFIG, CLIENT).build_index(
            faq_rows, paraphrase_rows)
    print(f"[app]   index built in {time.perf_counter() - t0:.1f}s")


def _warmup():
    print(f"[app] warming up {CONFIG['models']['primary']}...")
    try:
        CLIENT.warmup(CONFIG["models"]["primary"])
        print("[app]   warmup done")
    except LLMUnavailable as e:
        print(f"[app]   warmup failed: {e}")


def _init():
    faq_rows = _load_csv(CONFIG["paths"]["faq"])
    answer_bank_rows = _load_csv(CONFIG["paths"]["answer_bank"])
    idk_rows = _load_csv(CONFIG["paths"]["idk_templates"])

    global FAQ_LOOKUP, ANSWER_BANK, IDK_BANK
    FAQ_LOOKUP = _load_faq_lookup(faq_rows)
    ANSWER_BANK = _load_answer_bank(answer_bank_rows)
    IDK_BANK = _load_idk(idk_rows)

    _warmup()
    _connect_storage()
    _build_retriever()


# ---------- trace rendering (HTML cards for the drawer) -----------------

def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _chip(text, tone="neutral") -> str:
    return f'<span class="chip chip-{tone}">{_e(text)}</span>'


def _kv(key, value_html) -> str:
    return (f'<div class="tr-row"><span class="tr-k">{_e(key)}</span>'
            f'<span class="tr-v">{value_html}</span></div>')


def _bar(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    return f'<div class="bar"><span style="width:{pct:.0f}%"></span></div>'


def _section(title, body_html) -> str:
    return (f'<section class="tr-sec"><h4>{_e(title)}</h4>'
            f'{body_html}</section>')


TRACE_EMPTY = ('<div class="tr-empty">Send a question to see how '
               'the answer was produced.</div>')
TRACE_PENDING = '<div class="tr-empty">Processing...</div>'


def _format_trace(reply) -> str:
    t = reply.trace
    parts = []

    parts.append(
        '<div class="tr-summary">'
        + _chip(reply.path, "neutral")
        + _chip(f"{reply.latency_s:.2f}s", "neutral")
        + "</div>"
    )

    parts.append(_section("Input", "".join([
        _kv("Question", _e(t.query)),
        _kv("Normalized", f'<code>{_e(t.normalized)}</code>'),
        _kv("Language", _chip(t.language)),
    ])))

    if t.policy_handoff:
        policy = _chip("Handoff", "warn") + f' <code>{_e(t.policy_rule)}</code>'
    else:
        policy = _chip("Passed", "ok")
    parts.append(_section("Policy", _kv("Result", policy)))

    if t.retrieval_top:
        top = max((r["fused"] for r in t.retrieval_top), default=0) or 1
        rows = []
        for r in t.retrieval_top:
            faq = FAQ_LOOKUP.get(r["faq_id"], {})
            q = faq.get("question", "")
            q = q if len(q) <= 70 else q[:70] + "..."
            rows.append(
                '<div class="tr-hit">'
                f'<div class="tr-hit-head"><span>FAQ #{_e(r["faq_id"])}</span>'
                f'<span class="tr-num">{r["fused"]:.3f}</span></div>'
                f'{_bar(r["fused"] / top * 100)}'
                f'<div class="tr-hit-q">{_e(q)}</div>'
                '</div>'
            )
        parts.append(_section("Retrieval", "".join(rows)))

    parts.append(_section("Gate", "".join([
        _kv("Zone", _chip(t.gate_zone)),
        _kv("Verifier", _chip(t.gate_verifier)),
    ])))

    if t.stages:
        slowest = max((s.duration_s for s in t.stages), default=0) or 1
        rows = []
        for s in t.stages:
            rows.append(
                '<div class="tr-stage">'
                f'<div class="tr-hit-head"><code>{_e(s.name)}</code>'
                f'<span class="tr-num">{s.duration_s:.3f}s</span></div>'
                f'{_bar(s.duration_s / slowest * 100)}'
                '</div>'
            )
        parts.append(_section("Timing", "".join(rows)))

    return "".join(parts)


def _format_meta(reply) -> str:
    t = reply.trace
    path = str(reply.path).lower()
    handed_off = bool(t.policy_handoff) or "idk" in path or "handoff" in path
    text = f"Answered in {reply.latency_s:.1f}s"
    if handed_off:
        text += "  ·  Passed to a human agent"
    return text


# ---------- per-query handlers -----------------------------------------

THINKING = 'Thinking<span class="nb-dots"><i></i><i></i><i></i></span>'


def submit_handler(question, history):
    """Generator: first yield shows a pending reply, second the real one.

    Outputs: chatbot, question box, trace panel, meta line, send button.
    """
    history = history or []
    q = (question or "").strip()
    noop = gr.update()

    if not q:
        yield history, noop, noop, noop, noop
        return

    if RETRIEVER is None:
        msg = "Still warming up. Please try again in a few seconds."
        yield (history + [{"role": "user", "content": q},
                          {"role": "assistant", "content": msg}],
               gr.update(value=""), TRACE_EMPTY, "", noop)
        return

    pending = history + [
        {"role": "user", "content": q},
        {"role": "assistant", "content": THINKING},
    ]
    yield (pending,
           gr.update(value="", interactive=False),
           TRACE_PENDING,
           "",
           gr.update(interactive=False))

    try:
        reply = handle_query(
            q, RETRIEVER, CLIENT, CONFIG,
            FAQ_LOOKUP, ANSWER_BANK, IDK_BANK, storage=STORAGE)
    except Exception as e:
        final = history + [
            {"role": "user", "content": q},
            {"role": "assistant",
             "content": "Sorry, something went wrong. Please try again."},
        ]
        yield (final,
               gr.update(value="", interactive=True),
               f'<div class="tr-empty">{_e(type(e).__name__)}: {_e(e)}</div>',
               "",
               gr.update(interactive=True))
        return

    final = history + [
        {"role": "user", "content": q},
        {"role": "assistant", "content": reply.text},
    ]
    yield (final,
           gr.update(value="", interactive=True),
           _format_trace(reply),
           _format_meta(reply),
           gr.update(interactive=True))


def clear_handler():
    return [], TRACE_EMPTY, "", gr.update(value="", interactive=True)


# ---------- static content & generated assets ---------------------------

# Brand mark: burnt-orange tile with the NB monogram. Reused for the top
# bar, the welcome screen and the bot avatar so the brand stays consistent.
LOGO_TILE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">'
    '<defs><linearGradient id="orb" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#a5f3fc"/>'
    '<stop offset="1" stop-color="#0891b2"/>'
    '</linearGradient></defs>'
    '<circle cx="48" cy="48" r="34" fill="none" stroke="url(#orb)" '
    'stroke-width="5" stroke-linecap="round" stroke-dasharray="168 46" '
    'transform="rotate(-38 48 48)"/>'
    '<circle cx="48" cy="48" r="19" fill="none" stroke="#67e8f9" '
    'stroke-width="2" opacity=".48"/>'
    '<circle cx="48" cy="48" r="5" fill="#a5f3fc"/>'
    '<circle cx="75" cy="27" r="4" fill="#22d3ee"/>'
    '</svg>'
)

USER_AVATAR_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">'
    '<rect width="96" height="96" rx="26" fill="#1b1b21"/>'
    '<rect x="2.5" y="2.5" width="91" height="91" rx="24" fill="none" '
    'stroke="#33333c" stroke-width="3"/>'
    '<circle cx="48" cy="38" r="13" fill="#8f8a82"/>'
    '<path d="M24 78c4-15 44-15 48 0v4H24z" fill="#8f8a82"/>'
    '</svg>'
)

# Small line icons for the buttons (Feather-style strokes).
ICON_SEND_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="#ffffff" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4z"/>'
    '</svg>'
)
ICON_TRACE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="#a8a29e" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/>'
    '<line x1="12" y1="11" x2="12" y2="16.5"/>'
    '<line x1="12" y1="7.5" x2="12.01" y2="7.5"/>'
    '</svg>'
)
ICON_NEW_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="#a8a29e" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 5v14M5 12h14"/>'
    '</svg>'
)


def _ensure_asset_dir() -> Path:
    """A writable folder for generated assets that Gradio can serve.

    Prefers a folder next to app.py (always inside the app's working
    tree); falls back to the system temp directory if that is read-only.
    """
    local = Path(__file__).resolve().parent / "assets"
    try:
        local.mkdir(parents=True, exist_ok=True)
        probe = local / ".probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return local
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "nb_assist_assets"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


ASSET_DIR = _ensure_asset_dir()


def _asset(filename: str, svg: str) -> str:
    """Write an SVG asset; returns '' (no asset) if the disk write fails."""
    try:
        (ASSET_DIR / filename).write_text(svg, encoding="utf-8")
    except OSError:
        return ""
    return str(ASSET_DIR / filename)


BOT_AVATAR = _asset("nb-avatar-bot.svg",
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">'
    '<circle cx="48" cy="48" r="43" fill="#10232a" stroke="#164e63" stroke-width="2"/>'
    '<path d="M29 49a19 19 0 0 1 38 0v9H29z" fill="none" stroke="#67e8f9" stroke-width="4"/>'
    '<circle cx="40" cy="46" r="3" fill="#a5f3fc"/><circle cx="56" cy="46" r="3" fill="#a5f3fc"/>'
    '<path d="M40 65h16" stroke="#22d3ee" stroke-width="3" stroke-linecap="round"/>'
    '</svg>')
USER_AVATAR = _asset("nb-avatar-user.svg", USER_AVATAR_SVG)
ICON_SEND = _asset("nb-icon-send.svg", ICON_SEND_SVG)
ICON_TRACE = _asset("nb-icon-trace.svg", ICON_TRACE_SVG)
ICON_NEW = _asset("nb-icon-new.svg", ICON_NEW_SVG)

AVATAR_IMAGES = ((USER_AVATAR, BOT_AVATAR)
                 if USER_AVATAR and BOT_AVATAR else None)


SAMPLES = [
    "How do I reset my card PIN?",
    "amar card hariye gese, ekhon ki korbo?",
    "আমার কার্ড হারিয়ে গেছে, এখন কী করব?",
    "debit card er annual fee koto?",
    "Amar boyosh 17, ami ki student loan pabo?",
    "What is today's dollar to taka exchange rate?",
    "Do you offer cryptocurrency trading accounts?",
    "আজকের স্বর্ণের দাম কত?",
]
VISIBLE_SAMPLES = 4

WELCOME = """
<div class="nb-welcome">
  <h3>How can we help<br><span>you today?</span></h3>
  <p>Ask about cards, accounts or loans — we're here to help.</p>
</div>
"""

BRAND_HTML = """
<div class="nb-brand">
  <div class="nb-names">
    <div class="wm"><span class="wm-nb">NB</span> <span class="wm-assist">Assist</span></div>
    <div class="nb-sub">Banking FAQ Assistant</div>
  </div>
</div>
"""

FOOTER_HTML = """
<div class="nb-footnote">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M12 3l7 3v5c0 4.6-3 7.7-7 9-4-1.3-7-4.4-7-9V6z"/>
    <path d="M9 12l2 2 4-4"/>
  </svg>
  <span>Runs fully offline.
  NB Assist can make mistakes; for anything urgent, visit a branch or call
  the official hotline.</span>
</div>
"""

SAMPLES_LABEL_HTML = """
<div class="nb-samples-head">
  <div class="nb-samples-title">Sample questions</div>
  <div class="nb-samples-hint">Tap one to try it. A few are outside the FAQ, to show the human handoff.</div>
</div>
"""

DRAWER_TITLE_HTML = """
<div class="drawer-title">
  <span class="drawer-eyebrow">Under the hood</span>
  <span class="drawer-name">Decision trace</span>
</div>
"""

SCRIM_HTML = '<div id="nb-scrim"></div>'

TOGGLE_JS = """() => {
  const d = document.getElementById('trace-drawer');
  if (!d) return;
  const open = !d.classList.contains('open');
  d.classList.toggle('open', open);
  const s = document.getElementById('nb-scrim');
  if (s) s.classList.toggle('open', open);
  const b = document.getElementById('details-btn');
  if (b) b.classList.toggle('active', open);
}"""

CLOSE_JS = """() => {
  ['trace-drawer', 'nb-scrim'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove('open');
  });
  const b = document.getElementById('details-btn');
  if (b) b.classList.remove('active');
}"""

LOAD_JS = """() => {
  document.body.classList.add('dark');
  document.documentElement.classList.add('dark');
  if (window.__nbBound) return;
  window.__nbBound = true;
  const close = () => {
    ['trace-drawer', 'nb-scrim'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.remove('open');
    });
    const b = document.getElementById('details-btn');
    if (b) b.classList.remove('active');
  };
  document.addEventListener('click', e => {
    if (e.target && e.target.id === 'nb-scrim') close();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') close();
  });
}"""

FOCUS_JS = """() => {
  setTimeout(() => {
    const t = document.querySelector('#composer textarea');
    if (t) t.focus();
  }, 60);
}"""


# ---------- CSS ---------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root, .dark, body, .gradio-container {
  --nb-font: "DM Sans", "Noto Sans Bengali", "Nirmala UI", "Vrinda", sans-serif;
  --nb-display: "Space Grotesk", "DM Sans", sans-serif;
  --nb-mono: "JetBrains Mono", Consolas, monospace;

  --nb-white: #f5f4f7;
  --nb-muted: #b3b0bd;
  --nb-subtle: #858392;
  --nb-line: rgba(235, 232, 246, .17);
  --nb-line-strong: rgba(245, 243, 250, .30);
  --nb-glass: rgba(9, 13, 24, .42);
  --nb-glass-soft: rgba(13, 17, 31, .34);
  --nb-ease: cubic-bezier(.22, .8, .25, 1);

  color-scheme: dark;
  --body-background-fill: #070a12;
  --background-fill-primary: rgba(12, 16, 29, .78);
  --background-fill-secondary: rgba(20, 23, 37, .78);
  --block-background-fill: rgba(12, 16, 29, .48);
  --panel-background-fill: rgba(12, 16, 29, .48);
  --block-border-color: var(--nb-line);
  --border-color-primary: var(--nb-line);
  --input-background-fill: transparent;
  --body-text-color: var(--nb-white);
  --body-text-color-subdued: var(--nb-muted);
  --button-secondary-background-fill: rgba(255,255,255,.035);
  --button-secondary-background-fill-hover: #f5f4f7;
  --button-secondary-text-color: var(--nb-white);
  --button-secondary-border-color: var(--nb-line-strong);
  --color-accent: #d9d6e3;
  --color-accent-soft: rgba(255,255,255,.08);
  --link-text-color: #e8e5ef;
  --link-text-color-hover: #ffffff;
}

* { scrollbar-width: thin; scrollbar-color: rgba(220,218,232,.24) transparent; }
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(220,218,232,.20);
  border: 3px solid transparent;
  background-clip: content-box;
  border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(240,238,248,.36); background-clip: content-box; }

html, body, gradio-app {
  min-height: 100%;
  background: #070a12 !important;
}
body {
  background:
    radial-gradient(ellipse 58% 54% at 51% 42%, rgba(164,76,157,.31), transparent 72%),
    radial-gradient(ellipse 48% 54% at 17% 48%, rgba(85,68,119,.28), transparent 70%),
    radial-gradient(ellipse 52% 62% at 88% 43%, rgba(24,61,96,.43), transparent 72%),
    radial-gradient(ellipse 70% 40% at 50% 0%, rgba(20,48,76,.30), transparent 74%),
    #070a12 !important;
  background-attachment: fixed !important;
  color: var(--nb-white);
}
body::before {
  content: "";
  position: fixed;
  inset: -12%;
  z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(ellipse 42% 35% at 54% 42%, rgba(164,76,157,.22), transparent 74%),
    radial-gradient(ellipse 34% 45% at 86% 47%, rgba(34,91,130,.20), transparent 76%),
    radial-gradient(ellipse 38% 42% at 12% 55%, rgba(69,57,112,.18), transparent 76%);
  filter: blur(58px);
  opacity: .9;
}
footer { display: none !important; }
.gradio-container, .gradio-container *:not(code):not(pre) {
  font-family: var(--nb-font) !important;
}
.gradio-container {
  position: relative;
  z-index: 1;
  max-width: 1120px !important;
  min-height: 100vh;
  margin: 0 auto !important;
  padding: 28px 30px 26px !important;
  background: transparent !important;
}
.gradio-container > .main,
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain {
  background: transparent !important;
}
::selection { background: rgba(164,76,157,.42); color: #fff; }
button:focus-visible, textarea:focus-visible {
  outline: 2px solid rgba(245,243,250,.8) !important;
  outline-offset: 3px !important;
}
.nb-bare {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  min-height: 0 !important;
}

/* Motion */
@keyframes nb-rise {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes nb-message {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes nb-dot {
  0%, 80%, 100% { opacity: .35; transform: translateY(0); }
  40% { opacity: 1; transform: translateY(-4px); }
}
@keyframes nb-grow {
  from { transform: scaleX(0); }
  to { transform: scaleX(1); }
}

/* Header and wordmark */
#topbar {
  align-items: center !important;
  flex-wrap: nowrap !important;
  gap: 10px !important;
  margin: 0 0 15px !important;
  animation: nb-rise .55s var(--nb-ease) backwards;
}
#topbar > .block:first-child { flex: 1 1 auto !important; }
.nb-brand { display: flex; align-items: center; min-width: 0; }
.nb-names { min-width: 0; }
.wm {
  color: var(--nb-white);
  font-family: var(--nb-display) !important;
  font-size: 24px;
  font-weight: 600;
  letter-spacing: -1.1px;
  line-height: 1.05;
  white-space: nowrap;
}
.wm-nb, .wm-assist { color: var(--nb-white); font-weight: 600; }
.nb-sub {
  margin-top: 5px;
  color: var(--nb-subtle);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 1.45px;
  line-height: 1.3;
  text-transform: uppercase;
}
/* Explicitly size the Gradio button wrappers and their inner buttons.
   This prevents the New Chat label from wrapping into a narrow pill. */
#topbar #details-btn,
#topbar #new-chat-btn {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: max-content !important;
}
#topbar #details-btn button,
#topbar #new-chat-btn button,
#trace-drawer #close-trace button,
#topbar button,
#close-trace button {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 88px !important;
  height: 42px !important;
  padding: 0 17px !important;
  white-space: nowrap !important;
  word-break: keep-all !important;
  overflow-wrap: normal !important;
  border: 1px solid var(--nb-line-strong) !important;
  border-radius: 12px !important;
  background: rgba(255,255,255,.025) !important;
  color: var(--nb-white) !important;
  box-shadow: none !important;
  font-size: 12.5px !important;
  line-height: 1 !important;
  font-weight: 600 !important;
  transition: background .24s ease, color .24s ease, border-color .24s ease,
              transform .24s var(--nb-ease), box-shadow .24s ease !important;
}
#topbar #details-btn button { min-width: 88px !important; }
#topbar #new-chat-btn button { min-width: 104px !important; }
#trace-drawer #close-trace button {
  min-width: 88px !important;
  height: 42px !important;
  padding: 0 17px !important;
  border: 1px solid var(--nb-line-strong) !important;
  border-radius: 12px !important;
  background: rgba(255,255,255,.025) !important;
  color: var(--nb-white) !important;
  box-shadow: none !important;
}
#topbar #details-btn button:hover,
#topbar #new-chat-btn button:hover,
#topbar button:hover,
#close-trace button:hover {
  background: #f5f4f7 !important;
  color: #0a1020 !important;
  border-color: #f5f4f7 !important;
  box-shadow: 0 7px 22px rgba(0,0,0,.16) !important;
  transform: translateY(-1px);
}
/* Match the header controls: dark surface and light text, including on hover. */
#trace-drawer #close-trace button:hover {
  background: rgba(255,255,255,.10) !important;
  color: var(--nb-white) !important;
  border-color: rgba(245,243,250,.48) !important;
  box-shadow: 0 7px 22px rgba(0,0,0,.16) !important;
  transform: translateY(-1px);
}
#topbar button:active, #close-trace button:active { transform: scale(.97); }
#details-btn.active button {
  background: #f5f4f7 !important;
  color: #0a1020 !important;
  border-color: #f5f4f7 !important;
}

/* Main chat glass panel */
#chat-card {
  min-height: 0 !important;
  border: 1px solid rgba(238,235,247,.19) !important;
  border-radius: 25px !important;
  padding: 17px !important;
  background: linear-gradient(145deg, rgba(17,20,34,.42), rgba(7,11,22,.34)) !important;
  box-shadow: 0 28px 90px rgba(0,0,0,.28),
              inset 0 1px 0 rgba(255,255,255,.055) !important;
  backdrop-filter: blur(28px) saturate(125%) !important;
  -webkit-backdrop-filter: blur(28px) saturate(125%) !important;
  animation: nb-rise .6s var(--nb-ease) .06s backwards;
}
#chat-card .wrap,
#chat-card .bubble-wrap,
#chat-card .message-wrap,
#chat-card .placeholder,
#chat-card .empty,
#chat-card .message-container,
#chat-card [data-testid="user"],
#chat-card [data-testid="bot"] {
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
}
#chat-card .message,
#chat-card .message-body p,
#chat-card .message-wrap {
  font-size: 14.5px !important;
  line-height: 1.72 !important;
}
#chat-card .message-row {
  padding: 5px 0;
  animation: nb-message .32s var(--nb-ease) backwards;
}
#chat-card .message.user {
  background: rgba(245,243,250,.105) !important;
  color: var(--nb-white) !important;
  border: 1px solid rgba(245,243,250,.17) !important;
  border-radius: 17px 17px 5px 17px !important;
  box-shadow: 0 8px 24px rgba(0,0,0,.13) !important;
}
#chat-card .message.user * { color: var(--nb-white) !important; }
#chat-card .message.bot {
  background: rgba(8,12,23,.39) !important;
  color: var(--nb-white) !important;
  border: 1px solid rgba(235,232,246,.12) !important;
  border-radius: 5px 17px 17px 17px !important;
  box-shadow: 0 8px 24px rgba(0,0,0,.12) !important;
  backdrop-filter: blur(12px);
}
#chat-card .avatar-container img {
  width: 34px !important;
  height: 34px !important;
  border-radius: 50% !important;
  border: 1px solid rgba(235,232,246,.18);
  box-shadow: 0 4px 14px rgba(0,0,0,.22);
}
#chat-card .icon-button {
  color: #aaa7b4 !important;
  border-radius: 8px !important;
  transition: color .2s ease, background .2s ease;
}
#chat-card .icon-button:hover {
  color: #fff !important;
  background: rgba(255,255,255,.09) !important;
}
#chat-card .icon-button-wrapper.top-panel { display: none !important; }

/* Welcome state */
.nb-welcome {
  max-width: 580px;
  margin: 0 auto;
  padding: 28px 20px 24px;
  text-align: center;
  background: transparent !important;
}
.welcome-eyebrow {
  margin-bottom: 17px;
  color: #d2cbdc;
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 2.2px;
}
.nb-welcome h3 {
  margin: 0 0 12px !important;
  color: #fbfaff;
  font-family: var(--nb-display) !important;
  font-size: clamp(34px, 4.2vw, 48px) !important;
  font-weight: 500 !important;
  line-height: 1.08 !important;
  letter-spacing: -2.1px !important;
}
.nb-welcome h3 span {
  color: #e1d9e8;
  text-shadow: 0 0 34px rgba(164,76,157,.20);
}
.nb-welcome p {
  margin: 0 !important;
  color: #c0bdc9;
  font-size: 14px;
  line-height: 1.7;
}
.nb-lang-pills, .lang-pill, .nb-status { display: none !important; }
.nb-dots { display: inline-flex; gap: 4px; margin-left: 6px; vertical-align: middle; }
.nb-dots i {
  width: 6px; height: 6px; border-radius: 50%;
  background: #e7e3ed; opacity: .35;
  animation: nb-dot 1.2s infinite ease-in-out;
}
.nb-dots i:nth-child(2) { animation-delay: .15s; }
.nb-dots i:nth-child(3) { animation-delay: .30s; }
#meta-line {
  min-height: 14px;
  margin: 3px 6px 0 !important;
  color: var(--nb-muted);
  font-size: 11.5px;
  text-align: center;
}
#meta-line p { margin: 0 !important; }

/* Sample prompts — directly below the composer */
.nb-samples-head { padding: 13px 0 8px; text-align: center; }
.nb-samples-title {
  color: #e5e1eb;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.65px;
  text-transform: uppercase;
}
.nb-samples-hint {
  margin-top: 5px;
  color: var(--nb-subtle);
  font-size: 12px;
}
#suggestions, #suggestions-more {
  flex-wrap: wrap !important;
  justify-content: center !important;
  gap: 7px !important;
  margin-top: 0 !important;
  padding: 0 !important;
}
.chip-btn {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
  height: 34px !important;
  padding: 0 14px !important;
  border: 1px solid rgba(235,232,246,.22) !important;
  border-radius: 999px !important;
  background: rgba(9,13,24,.20) !important;
  color: #e1dee8 !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  white-space: nowrap !important;
  box-shadow: none !important;
  transition: background .24s ease, color .24s ease, border-color .24s ease,
              transform .24s var(--nb-ease), box-shadow .24s ease !important;
}
.chip-btn:hover {
  background: #f5f4f7 !important;
  color: #0a1020 !important;
  border-color: #f5f4f7 !important;
  box-shadow: 0 6px 18px rgba(0,0,0,.14) !important;
  transform: translateY(-1px);
}
.chip-btn:active { transform: scale(.97); }
#more-samples {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
#more-samples > .label-wrap {
  justify-content: center !important;
  padding: 5px 0 !important;
  color: #c4c0cc !important;
  font-size: 12px !important;
}
#more-samples > .label-wrap:hover { color: #fff !important; }

/* Composer */
#composer {
  align-items: center !important;
  flex-wrap: nowrap !important;
  gap: 10px !important;
  margin-top: 9px !important;
  margin-bottom: 0 !important;
  padding: 7px 7px 7px 18px !important;
  border: 1px solid rgba(235,232,246,.21) !important;
  border-radius: 17px !important;
  background: rgba(8,12,23,.43) !important;
  box-shadow: 0 14px 38px rgba(0,0,0,.20),
              inset 0 1px 0 rgba(255,255,255,.035) !important;
  backdrop-filter: blur(22px);
  -webkit-backdrop-filter: blur(22px);
  transition: border-color .22s ease, box-shadow .22s ease;
}
#composer:focus-within {
  border-color: rgba(245,243,250,.48) !important;
  box-shadow: 0 0 0 3px rgba(255,255,255,.055),
              0 14px 38px rgba(0,0,0,.20) !important;
}
#composer textarea {
  min-width: 0 !important;
  padding: 10px 4px !important;
  border: 0 !important;
  outline: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  color: #f5f4f7 !important;
  font-size: 14px !important;
}
#composer textarea::placeholder { color: #9693a1 !important; opacity: 1 !important; }
#composer .block, #composer .form {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
#send-btn {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 92px !important;
  height: 42px !important;
  padding: 0 18px !important;
  border: 1px solid #f5f4f7 !important;
  border-radius: 12px !important;
  background: #f5f4f7 !important;
  color: #0a1020 !important;
  box-shadow: 0 5px 18px rgba(0,0,0,.14) !important;
  font-size: 12.5px !important;
  font-weight: 700 !important;
  transition: background .24s ease, color .24s ease, border-color .24s ease,
              transform .24s var(--nb-ease), box-shadow .24s ease !important;
}
#send-btn:hover {
  background: #ffffff !important;
  color: #070a12 !important;
  border-color: #ffffff !important;
  box-shadow: 0 8px 24px rgba(0,0,0,.22) !important;
  transform: translateY(-1px);
}
#send-btn:active { transform: scale(.97); }
#send-btn:disabled {
  background: rgba(245,244,247,.72) !important;
  border-color: rgba(245,244,247,.72) !important;
  color: rgba(10,16,32,.72) !important;
  opacity: 1 !important;
  cursor: progress;
}

/* Trust note */
.nb-footnote {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  gap: 8px;
  max-width: 650px;
  margin: 15px auto 0;
  color: #aaa6b3;
  font-size: 11px;
  line-height: 1.6;
  text-align: left;
}
.nb-footnote svg {
  flex: 0 0 15px;
  width: 15px;
  height: 15px;
  margin-top: 2px;
  color: #d5d1dc;
}

/* Decision trace drawer */
#nb-scrim {
  position: fixed;
  inset: 0;
  z-index: 1000;
  background: rgba(2,4,10,.58);
  backdrop-filter: blur(5px);
  opacity: 0;
  pointer-events: none;
  transition: opacity .28s ease;
}
#nb-scrim.open { opacity: 1; pointer-events: auto; }
#trace-drawer {
  position: fixed !important;
  top: 0 !important; right: 0 !important; bottom: 0 !important;
  width: min(440px, 92vw) !important;
  min-width: 0 !important; max-width: 92vw !important;
  height: 100vh !important;
  margin: 0 !important;
  padding: 22px 20px 30px !important;
  z-index: 1001 !important;
  gap: 12px !important;
  overflow-x: hidden !important; overflow-y: auto !important;
  border: 1px solid rgba(235,232,246,.18) !important;
  border-right: 0 !important;
  border-radius: 20px 0 0 20px !important;
  background: rgba(8,11,21,.91) !important;
  box-shadow: -18px 0 55px rgba(0,0,0,.38) !important;
  backdrop-filter: blur(28px) saturate(125%) !important;
  -webkit-backdrop-filter: blur(28px) saturate(125%) !important;
  transform: translateX(108%);
  visibility: hidden;
  transition: transform .34s var(--nb-ease), visibility 0s linear .34s;
}
#trace-drawer.open {
  transform: translateX(0);
  visibility: visible;
  transition: transform .34s var(--nb-ease), visibility 0s;
}
#trace-drawer > * { min-width: 0 !important; max-width: 100% !important; }
#drawer-head {
  align-items: center !important;
  flex-wrap: nowrap !important;
  justify-content: space-between !important;
}
.drawer-title { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.drawer-eyebrow {
  color: #aaa6b5;
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 1.55px;
  text-transform: uppercase;
}
.drawer-name {
  color: #f5f4f7;
  font-family: var(--nb-display) !important;
  font-size: 19px;
  font-weight: 600;
  letter-spacing: -.45px;
}
#close-trace { height: 33px !important; padding: 0 14px !important; font-size: 12px !important; }
#trace-body, #trace-body * { overflow-wrap: anywhere; word-break: break-word; }
.tr-empty {
  padding: 30px 8px;
  color: var(--nb-muted);
  font-size: 13px;
  text-align: center;
}
.tr-summary { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 12px; }
.tr-sec {
  padding: 13px 14px;
  margin-bottom: 9px;
  border: 1px solid rgba(235,232,246,.13);
  border-radius: 14px;
  background: rgba(255,255,255,.035);
  animation: nb-rise .35s var(--nb-ease) backwards;
}
.tr-sec h4 {
  margin: 0 0 8px;
  color: #e2deea;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.1px;
  text-transform: uppercase;
}
.tr-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  padding: 5px 0;
  font-size: 12.5px;
}
.tr-k { flex: 0 0 auto; color: #aaa6b3; }
.tr-v { min-width: 0; color: #f0eef4; text-align: right; }
.tr-hit, .tr-stage { padding: 6px 0; }
.tr-hit-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
  color: #e8e5ed;
  font-size: 12.5px;
  font-weight: 600;
}
.tr-num { color: #e5e1eb; font-family: var(--nb-mono) !important; font-size: 11.5px; }
.tr-hit-q { margin-top: 6px; color: #aaa6b3; font-size: 11.5px; line-height: 1.5; }
.bar { height: 5px; overflow: hidden; border-radius: 999px; background: rgba(255,255,255,.10); }
.bar span {
  display: block;
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #756b88, #d4c6d9);
  transform-origin: left center;
  animation: nb-grow .65s var(--nb-ease) .12s backwards;
}
#trace-body code {
  padding: 2px 6px;
  border: 1px solid rgba(235,232,246,.13);
  border-radius: 6px;
  background: rgba(255,255,255,.055);
  color: #eeeaf3;
  font-family: var(--nb-mono) !important;
  font-size: 11px;
}
.chip {
  display: inline-block;
  padding: 3px 9px;
  border: 1px solid rgba(235,232,246,.15);
  border-radius: 999px;
  background: rgba(255,255,255,.06);
  color: #e7e3ec;
  font-size: 11px;
  font-weight: 600;
}
.chip-neutral { background: rgba(255,255,255,.06); color: #e7e3ec; }
.chip-ok { background: rgba(52,211,153,.10); color: #8fe0bf; border-color: rgba(52,211,153,.18); }
.chip-warn { background: rgba(251,191,36,.10); color: #f0d28a; border-color: rgba(251,191,36,.18); }

@media (max-width: 720px) {
  .gradio-container { padding: 18px 14px 22px !important; }
  #topbar { margin-bottom: 15px !important; }
  .wm { font-size: 21px; }
  .nb-sub { font-size: 8.5px; letter-spacing: 1px; }
  #topbar button { min-width: 0 !important; height: 38px !important; padding: 0 11px !important; font-size: 11.5px !important; }
  #topbar #details-btn button { min-width: 76px !important; }
  #topbar #new-chat-btn button { min-width: 88px !important; }
  #chat-card { min-height: 0 !important; padding: 13px !important; border-radius: 20px !important; }
  .nb-welcome { padding: 28px 12px 24px; }
  .nb-welcome h3 { font-size: 36px !important; }
  #suggestions, #suggestions-more {
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    justify-content: flex-start !important;
    min-width: 0 !important;
    contain: inline-size;
    padding: 3px 2px 7px !important;
    scrollbar-width: none;
  }
  #suggestions::-webkit-scrollbar, #suggestions-more::-webkit-scrollbar { display: none; }
  .nb-samples-hint { max-width: 440px; margin-left: auto; margin-right: auto; }
  #composer { padding-left: 12px !important; }
  #send-btn { min-width: 74px !important; padding: 0 12px !important; }
  .nb-footnote { font-size: 10.5px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .01ms !important;
  }
}

/* FINAL OVERRIDE: Gradio assigns elem_id directly to the Close button.
   Target that element itself (not a child button) and its text descendants. */
#trace-drawer #close-trace,
#trace-drawer #close-trace:hover,
#trace-drawer #close-trace:focus,
#trace-drawer #close-trace:focus-visible,
#trace-drawer #close-trace:active,
#trace-drawer #close-trace[aria-expanded="true"] {
  background: rgba(255,255,255,.035) !important;
  background-color: rgba(255,255,255,.035) !important;
  background-image: none !important;
  color: #f5f4f7 !important;
  border: 1px solid rgba(245,243,250,.30) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  filter: none !important;
}
#trace-drawer #close-trace *,
#trace-drawer #close-trace:hover *,
#trace-drawer #close-trace:focus *,
#trace-drawer #close-trace:active * {
  color: #f5f4f7 !important;
  background: transparent !important;
  background-color: transparent !important;
  -webkit-text-fill-color: #f5f4f7 !important;
}
#trace-drawer #close-trace:hover {
  background: rgba(255,255,255,.10) !important;
  background-color: rgba(255,255,255,.10) !important;
  border-color: rgba(245,243,250,.48) !important;
  box-shadow: 0 7px 22px rgba(0,0,0,.16) !important;
  transform: translateY(-1px);
}
#trace-drawer #close-trace:active { transform: scale(.97); }

"""

# ---------- theme --------------------------------------------------------

THEME = (
    gr.themes.Soft(
        primary_hue="slate",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_lg,
    )
    .set(
        body_background_fill="#070a12",
        body_background_fill_dark="#070a12",
        background_fill_primary="#0c101d",
        background_fill_primary_dark="#0c101d",
        background_fill_secondary="#141725",
        background_fill_secondary_dark="#141725",
        block_background_fill="#0c101d",
        block_background_fill_dark="#0c101d",
        block_border_color="rgba(235,232,246,.17)",
        block_border_color_dark="rgba(235,232,246,.17)",
        body_text_color="#f5f4f7",
        body_text_color_dark="#f5f4f7",
        body_text_color_subdued="#b3b0bd",
        body_text_color_subdued_dark="#b3b0bd",
        button_primary_background_fill="#f5f4f7",
        button_primary_background_fill_dark="#f5f4f7",
        button_primary_background_fill_hover="#ffffff",
        button_primary_background_fill_hover_dark="#ffffff",
        button_primary_text_color="#0a1020",
        button_primary_text_color_dark="#0a1020",
        button_secondary_background_fill="rgba(255,255,255,.035)",
        button_secondary_background_fill_dark="rgba(255,255,255,.035)",
        button_secondary_background_fill_hover="#f5f4f7",
        button_secondary_background_fill_hover_dark="#f5f4f7",
        button_secondary_border_color="rgba(235,232,246,.22)",
        button_secondary_border_color_dark="rgba(235,232,246,.22)",
        button_secondary_text_color="#f5f4f7",
        button_secondary_text_color_dark="#f5f4f7",
        link_text_color="#e8e5ef",
        link_text_color_dark="#e8e5ef",
    )
)

# Injected into <head> before the app renders: kills the white flash on
# load and tells the browser we are a dark app (native form controls).
HEAD_HTML = (
    '<meta name="color-scheme" content="dark">'
    "<style>html{background:#050507}</style>"
)


# ---------- launch ------------------------------------------------------

def main():
    _init()
    print("[app] launching Gradio UI...")

    with gr.Blocks(title="NB Assist") as demo:
        # Top bar: brand lockup + window controls
        with gr.Row(elem_id="topbar"):
            gr.HTML(BRAND_HTML, elem_classes=["nb-bare"])
            details_btn = gr.Button(
                "Details", variant="secondary", scale=0,
                elem_id="details-btn")
            clear_btn = gr.Button(
                "New chat", variant="secondary", scale=0,
                elem_id="new-chat-btn")

        # Chat
        chatbot = gr.Chatbot(
            height="clamp(260px, calc(100dvh - 570px), 390px)",
            show_label=False,
            placeholder=WELCOME,
            autoscroll=True,
            avatar_images=AVATAR_IMAGES,
            buttons=["copy"],
            elem_id="chat-card",
        )
        meta_out = gr.Markdown("", elem_id="meta-line")

        # Composer
        with gr.Row(elem_id="composer"):
            question_in = gr.Textbox(
                placeholder="Ask about cards, accounts or loans…",
                show_label=False, container=False,
                scale=1, lines=1, max_lines=4,
                autofocus=True,
            )
            send_btn = gr.Button(
                "Send", variant="primary", scale=0,
                elem_id="send-btn")


        # Sample questions: four visible, the rest tucked away
        gr.HTML(SAMPLES_LABEL_HTML, elem_classes=["nb-bare"])
        chip_buttons = []
        with gr.Row(elem_id="suggestions"):
            for s in SAMPLES[:VISIBLE_SAMPLES]:
                chip_buttons.append(
                    gr.Button(s, size="sm", scale=0, elem_classes=["chip-btn"]))
        with gr.Accordion("More examples", open=False, elem_id="more-samples"):
            with gr.Row(elem_id="suggestions-more"):
                for s in SAMPLES[VISIBLE_SAMPLES:]:
                    chip_buttons.append(
                        gr.Button(s, size="sm", scale=0,
                                  elem_classes=["chip-btn"]))

        # Trust footnote
        gr.HTML(FOOTER_HTML, elem_classes=["nb-bare"])

        # Backdrop + slide-in trace drawer (positioned fixed)
        gr.HTML(SCRIM_HTML, elem_classes=["nb-bare"])
        with gr.Column(elem_id="trace-drawer", min_width=0):
            with gr.Row(elem_id="drawer-head"):
                gr.HTML(DRAWER_TITLE_HTML, elem_classes=["nb-bare"])
                close_btn = gr.Button("Close", size="sm",
                                      variant="secondary", scale=0,
                                      elem_id="close-trace")
            trace_out = gr.HTML(TRACE_EMPTY, elem_id="trace-body",
                                elem_classes=["nb-bare"])

        # Events
        submit_io = dict(
            fn=submit_handler,
            inputs=[question_in, chatbot],
            outputs=[chatbot, question_in, trace_out, meta_out, send_btn],
            show_progress="hidden",
        )

        def refocus(event):
            return event.then(fn=None, inputs=None, outputs=None, js=FOCUS_JS)

        refocus(send_btn.click(**submit_io))
        refocus(question_in.submit(**submit_io))

        for btn, sample in zip(chip_buttons, SAMPLES):
            refocus(
                btn.click(
                    fn=lambda s=sample: s,
                    inputs=None,
                    outputs=question_in,
                    show_progress="hidden",
                ).then(**submit_io)
            )

        details_btn.click(fn=None, inputs=None, outputs=None, js=TOGGLE_JS)
        close_btn.click(fn=None, inputs=None, outputs=None, js=CLOSE_JS)
        clear_btn.click(
            clear_handler,
            inputs=None,
            outputs=[chatbot, trace_out, meta_out, question_in],
        )

        demo.load(fn=None, inputs=None, outputs=None, js=LOAD_JS)

    demo.queue()
    demo.launch(
        server_name="127.0.0.1", server_port=7860, show_error=True,
        allowed_paths=[str(ASSET_DIR)],
        theme=THEME, css=CSS, head=HEAD_HTML,
    )


if __name__ == "__main__":
    main()
