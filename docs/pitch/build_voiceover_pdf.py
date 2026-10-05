"""Builds docs/pitch/voiceover_transcript.pdf: the 12 voice-over clips to record (one per paragraph), with what is
on screen. The text is imported from scripts/make_video.py, so what is read and what is captioned never drift.
Run: uv run --with reportlab python docs/pitch/build_voiceover_pdf.py"""
import importlib.util
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("make_video", ROOT / "scripts" / "make_video.py")
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)

OUT = Path(__file__).with_name("voiceover_transcript.pdf")
INK, MUTED, ACCENT = HexColor("#14213D"), HexColor("#6A7382"), HexColor("#B85E12")
title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=22, leading=28, textColor=INK, spaceAfter=8)
small = ParagraphStyle("s", fontName="Helvetica", fontSize=12, leading=17, textColor=INK, spaceAfter=6)
clip = ParagraphStyle("c", fontName="Helvetica-Bold", fontSize=15, leading=20, textColor=ACCENT, spaceBefore=4)
cue = ParagraphStyle("q", fontName="Helvetica-Oblique", fontSize=11, leading=15, textColor=MUTED, spaceAfter=6)
say = ParagraphStyle("y", fontName="Helvetica", fontSize=19, leading=29, textColor=INK, spaceAfter=20)
SCREEN = {
    "slide:cover": "Slide 1 · cover", "slide:why": "Slide 2 · why disputes",
    "demo:intro": "Live app, scenario list", "demo:auth": "Ambiguous scenario in Portuguese: document and SMS code",
    "demo:choose": "The agent lists the matching charges; the customer answers", "demo:question": "A question at the confirmation",
    "demo:write": "Dispute filed, card blocked, live trace", "demo:handoff": "Large amount: hand-off to a person",
    "slide:architecture": "Slide 3 · architecture", "slide:router": "Slide 4 · learned router",
    "slide:results": "Slide 5 · held-out results", "slide:production": "Slide 6 · route to production",
}

mv.SEGMENTS[0][1][0] = mv.HUMAN_OPENING
story = [Paragraph("Voice-over · 12 clips", title),
         Paragraph("Record each paragraph as its own clip and name the files <b>01</b> to <b>12</b> "
                   "(Voice Memos or QuickTime: File &gt; New Audio Recording; any format is fine). "
                   "If you stumble, record that clip again; nothing else needs redoing.", small),
         Paragraph("Speak calmly and leave half a second of silence at the start and end; I trim it and match "
                   "the video to the length of each clip, so there is no timing to follow.", small),
         Paragraph("Quiet room, phone or laptop mic 20 cm away, same distance for every clip.", small),
         Spacer(1, 10)]
for i, (kind, sentences) in enumerate(mv.SEGMENTS):
    if i in (2, 8):
        story.append(PageBreak())
    story.append(KeepTogether([Paragraph(f"Clip {i + 1:02d}", clip),
                               Paragraph(f"On screen: {SCREEN[kind]}", cue),
                               Paragraph(" ".join(sentences), say)]))
SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                  topMargin=0.75 * inch, bottomMargin=0.75 * inch, title="Voice-over transcript").build(story)
print(OUT)
