"""Builds the pitch video with no recording session: synthetic narration (macOS `say`), slides rendered with Chromium,
a live demo driven by Playwright against the deployed app, and burned-in captions, joined with ffmpeg.

Run: uv run --with playwright python scripts/make_video.py --deck <dir with project/slides/*.html> --out <dir>
Needs: macOS `say`, ffmpeg/ffprobe on PATH, Playwright's Chromium."""
import argparse
import json
import re
import subprocess
import time
from html import escape
from pathlib import Path

from playwright.sync_api import sync_playwright

APP = "https://latam-dispute-agent.onrender.com"
VOICE, RATE = "Samantha", 172
W, H, VW, VH, BAND = 1920, 1080, 1680, 945, 135   # final frame; content area; caption band height
BG = "#0d1424"
GAP, TAIL = 0.25, 0.6                             # silence between sentences / at the end of a segment

# Each segment: kind (slide id or demo step) and its narration, one caption per sentence.
SEGMENTS = [
    ("slide:cover", [
        "Hi. This is Diego's entry to the Factored AI and Data Hackathon, narrated by a synthetic voice.",
        "It is a dispute agent for LATAM Bank that knows when not to act.",
        "It handles card disputes end to end, in Spanish and Portuguese, and hands off to a person when policy says so."]),
    ("slide:why", [
        "We chose disputes from the data.",
        "Complaint contacts are fixed on first contact only 44 percent of the time, against 92 percent for transactional ones.",
        "Unrecognized charges and undue fees are about 40 percent of formal complaints.",
        "And every transaction ties cleanly to its product and customer, so the agent can verify the exact charge."]),
    ("demo:intro", [
        "Here it is, live, on the deployed app.",
        "The interface is in Spanish, because that is what LATAM Bank customers speak.",
        "On the right, demo scenarios with real sandbox customers. On top, a progress bar that follows the agent's real stage."]),
    ("demo:auth", [
        "A customer writes in Portuguese about a charge they don't recognize.",
        "The router detects Portuguese and a dispute.",
        "Before looking at any data, the agent verifies identity: first the document, then a one-time code, simulated here as an SMS."]),
    ("demo:choose", [
        "There are several charges at that merchant, so instead of guessing, it asks.",
        "The customer answers in their own words, and the model maps that to the right option."]),
    ("demo:question", [
        "Before any write, it asks for an explicit yes.",
        "A question is not a yes: it asks again, and writes nothing."]),
    ("demo:write", [
        "Now it files the dispute, verifies it by reading it back, and offers to block the card.",
        "The progress bar names the outcome.",
        "And the live trace shows every tool call, the policy rule, and the cost of each turn: fractions of a cent."]),
    ("demo:handoff", [
        "And when it should not act: a large withdrawal.",
        "After verifying identity, policy says a person must review it, so the agent hands off with a reference.",
        "The advisor receives everything already verified, and the progress bar marks the confirmation step as skipped."]),
    ("slide:architecture", [
        "The core design decision: the model understands, code decides.",
        "A learned router and an LLM read the message. The LLM never sees the document or the one-time code.",
        "A state machine, a versioned policy, and customer-scoped bank tools decide and act.",
        "When the router and the model disagree, the agent asks the customer instead of guessing."]),
    ("slide:router", [
        "The router is the learned component.",
        "It was chosen by a rule fixed before the sealed, hand-written test set was opened.",
        "Both learned routers beat the keyword baseline by more than 20 points of macro F1."]),
    ("slide:results", [
        "End to end, we ran 230 sealed conversations on fresh records through both systems.",
        "The hybrid safely resolves 98 percent of in-scope disputes, against 86.5 percent for rules.",
        "It makes every required handoff, and it has one unsafe outcome, which we report, along with how we got here."]),
    ("slide:production", [
        "It runs today, with tracing, bounded retries, a spending cap, and a safe fallback.",
        "And we list what is missing to make it real.",
        "Build something that works, prove it works, and know when not to act. Thank you."]),
]


def run(*cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(json.loads(out)["format"]["duration"])


def narrate(out: Path) -> dict:
    """One wav per sentence; returns {segment index: [(wav, seconds), ...]}."""
    audio = {}
    for i, (_, sentences) in enumerate(SEGMENTS):
        audio[i] = []
        for j, text in enumerate(sentences):
            aiff, wav = out / f"a{i:02d}_{j}.aiff", out / f"a{i:02d}_{j}.wav"
            run("say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), text)
            run("ffmpeg", "-y", "-i", str(aiff), "-ar", "48000", "-ac", "1", str(wav))
            audio[i].append((wav, duration(wav)))
    return audio


SLIDE_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>*{box-sizing:border-box;margin:0}body{width:1920px;height:1080px;overflow:hidden}
section{width:1920px;height:1080px;position:relative}aside{display:none}ul{padding-left:1.1em}
li{margin-bottom:10px}th,td{padding:.35em .6em;border-bottom:1px solid rgba(127,127,127,.35);text-align:left}
table{border-collapse:collapse}</style></head><body>__SECTION__</body></html>"""
LOCK = ('<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#F0A052" stroke-width="2">'
        '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>')


def slide_html(section: str) -> str:
    def arrow(m):
        style = m.group(1)
        return f'<div style="{style}; clip-path:polygon(0 30%,60% 30%,60% 0,100% 50%,60% 100%,60% 70%,0 70%); flex:none"></div>'
    section = re.sub(r'<x-shape kind="arrow-right" style="([^"]*)"></x-shape>', arrow, section)
    section = re.sub(r'<x-icon name="Lock"[^>]*></x-icon>', LOCK, section)
    return SLIDE_PAGE.replace("__SECTION__", section)


CAPTION_PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>*{margin:0}body{width:1920px;height:135px;
background:__BG__;display:flex;align-items:center;justify-content:center;font:600 30px/1.3 -apple-system,Helvetica,Arial,
sans-serif;color:#F6F4EE;text-align:center;padding:0 140px;box-sizing:border-box}</style></head><body>__TEXT__</body></html>"""


def render_stills(browser, deck: Path, out: Path):
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    for f in sorted((deck / "project" / "slides").glob("*.html")):
        page.set_content(slide_html(f.read_text()), wait_until="networkidle")
        page.screenshot(path=str(out / f"slide_{f.stem}.png"))
    page.set_viewport_size({"width": 1920, "height": BAND})
    for i, (_, sentences) in enumerate(SEGMENTS):
        for j, text in enumerate(sentences):
            page.set_content(CAPTION_PAGE.replace("__BG__", BG).replace("__TEXT__", escape(text)))
            page.screenshot(path=str(out / f"cap{i:02d}_{j}.png"))
    page.close()


# ---- live demo, recorded as one video with a timestamp per segment ------------------------------------------------
def record_demo(browser, out: Path) -> tuple[Path, dict]:
    ctx = browser.new_context(viewport={"width": 1536, "height": 864}, record_video_dir=str(out),
                              record_video_size={"width": 1536, "height": 864})
    page = ctx.new_page()
    t0 = time.monotonic()
    marks = {}
    now = lambda: time.monotonic() - t0  # noqa: E731

    def agent_count():
        return page.locator(".msg.agent").count()

    def wait_reply(before):
        page.wait_for_function(f"document.querySelectorAll('.msg.agent').length > {before} && "
                               "!document.getElementById('chat').classList.contains('typing')", timeout=90000)
        page.wait_for_timeout(1200)

    def click_chip(primary=True, text=None):
        before = agent_count()
        loc = page.locator("#next .chip.primary" if primary else f"#next .chip:has-text('{text}')").first
        loc.click()
        wait_reply(before)

    def type_send(text):
        before = agent_count()
        page.locator("#input").click()
        page.locator("#input").type(text, delay=55)
        page.wait_for_timeout(300)
        page.keyboard.press("Enter")
        wait_reply(before)

    def scenario(title, lang):
        before = agent_count()
        page.locator("#scenarios .scenario", has_text=title).first.locator(f"button:has-text('({lang})')").click()
        wait_reply(before)

    page.goto(APP, wait_until="networkidle")
    page.wait_for_selector("#scenarios .scenario button", timeout=180000)
    page.wait_for_timeout(1500)

    marks["demo:intro"] = [now()]
    page.wait_for_timeout(2500)
    marks["demo:intro"].append(now())

    marks["demo:auth"] = [now()]
    scenario("Cargo ambiguo", "PT")
    click_chip()                            # document
    page.wait_for_timeout(1200)             # let the SMS card be seen
    click_chip()                            # code
    marks["demo:auth"].append(now())

    marks["demo:choose"] = [now()]
    first = page.locator("#next .chip").first.inner_text()
    amount = re.search(r"(\d+)[.,]\d{2}", first)
    if amount:
        type_send(f"o de {amount.group(1)}")
    else:
        click_chip(primary=False, text="1 ·")
    marks["demo:choose"].append(now())

    marks["demo:question"] = [now()]
    type_send("o que acontece se eu confirmar?")
    marks["demo:question"].append(now())

    marks["demo:write"] = [now()]
    click_chip()                            # Sim, abrir contestação
    click_chip()                            # Sim, bloquear
    page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    page.wait_for_timeout(2500)
    page.evaluate("document.querySelector('#trace').scrollIntoView({behavior: 'smooth', block: 'center'})")
    page.wait_for_timeout(4000)
    page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    page.wait_for_timeout(1500)
    marks["demo:write"].append(now())

    marks["demo:handoff"] = [now()]
    page.locator("#next .chip.primary").click()     # try another scenario
    page.wait_for_timeout(2500)
    page.evaluate("window.scrollTo({top: 0})")
    scenario("Monto alto", "ES")
    click_chip()                            # document
    page.wait_for_timeout(800)
    click_chip()                            # code -> handoff
    page.locator(".handoff summary").click()
    page.wait_for_timeout(3500)
    marks["demo:handoff"].append(now())

    video = Path(page.video.path())
    ctx.close()                             # flushes the video file
    return video, marks


# ---- assembly ---------------------------------------------------------------------------------------------------
def build_segment(i, kind, audio, out: Path, demo_video: Path | None, marks: dict) -> Path:
    durs = [d for _, d in audio[i]]
    talk = sum(durs) + GAP * (len(durs) - 1) + TAIL
    inputs, total = [], talk
    if kind.startswith("slide:"):
        inputs += ["-loop", "1", "-framerate", "30", "-i", str(out / f"slide_{kind.split(':')[1]}.png")]
        base = "[0:v]scale=1680:945,setsar=1"
    else:
        start, end = marks[kind]
        total = max(talk, end - start)
        inputs += ["-ss", f"{start:.2f}", "-t", f"{end - start:.2f}", "-i", str(demo_video)]
        base = (f"[0:v]fps=30,scale=1680:945,setsar=1,"
                f"tpad=stop_mode=clone:stop_duration={max(0.0, total - (end - start)) + 0.5:.2f}")
    for j in range(len(durs)):
        inputs += ["-i", str(out / f"cap{i:02d}_{j}.png")]
    first_audio = 1 + len(durs)
    for wav, _ in audio[i]:
        inputs += ["-i", str(wav)]
    f = [f"{base},pad={W}:{H}:{(W - VW) // 2}:0:color={BG}[v0]"]
    t = 0.0
    for j, d in enumerate(durs):
        f.append(f"[v{j}][{1 + j}:v]overlay=0:{H - BAND}:enable='between(t,{t:.2f},{t + d + GAP:.2f})'[v{j + 1}]")
        t += d + GAP
    pieces = []
    for j in range(len(durs)):
        f.append(f"[{first_audio + j}:a]apad=pad_dur={GAP}[a{j}]")
        pieces.append(f"[a{j}]")
    f.append(f"{''.join(pieces)}concat=n={len(durs)}:v=0:a=1,apad[aout]")
    seg = out / f"seg{i:02d}.mp4"
    run("ffmpeg", "-y", *inputs, "-filter_complex", ";".join(f), "-map", f"[v{len(durs)}]", "-map", "[aout]",
        "-t", f"{total:.2f}", "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", str(seg))
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reuse-demo", type=Path, help="skip recording: reuse <out>/demo.webm and <out>/marks.json")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    print("narrating…", flush=True)
    audio = narrate(out)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        print("rendering slides and captions…", flush=True)
        render_stills(browser, args.deck, out)
        if args.reuse_demo:
            demo, marks = out / "demo.webm", json.loads((out / "marks.json").read_text())
        else:
            print("recording the live demo…", flush=True)
            raw, marks = record_demo(browser, out)
            demo = out / "demo.webm"
            raw.replace(demo)
            (out / "marks.json").write_text(json.dumps(marks))
        browser.close()
    print("assembling…", flush=True)
    segs = [build_segment(i, kind, audio, out, demo, marks) for i, (kind, _) in enumerate(SEGMENTS)]
    (out / "list.txt").write_text("".join(f"file '{s.name}'\n" for s in segs))
    final = out / "latam_dispute_agent_pitch.mp4"
    run("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(out / "list.txt"), "-c", "copy", str(final))
    print(f"{final}  {duration(final):.1f}s", flush=True)


if __name__ == "__main__":
    main()
