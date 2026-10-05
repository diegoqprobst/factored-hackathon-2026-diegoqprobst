"""Builds docs/pitch/video_script_print.pdf: the pitch narration in large type, to print and read while recording.
Run: uv run --with reportlab python docs/pitch/build_print_script.py"""
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).with_name("video_script_print.pdf")
INK, MUTED, ACCENT = HexColor("#14213D"), HexColor("#6A7382"), HexColor("#B85E12")

title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, leading=28, textColor=INK, spaceAfter=6)
small = ParagraphStyle("small", fontName="Helvetica", fontSize=11, leading=15, textColor=MUTED)
check = ParagraphStyle("check", fontName="Helvetica", fontSize=13, leading=19, textColor=INK, leftIndent=18,
                       bulletIndent=0, spaceAfter=6)
block = ParagraphStyle("block", fontName="Helvetica-Bold", fontSize=15, leading=20, textColor=ACCENT, spaceBefore=6,
                       spaceAfter=10)
cue = ParagraphStyle("cue", fontName="Helvetica-Oblique", fontSize=12, leading=16, textColor=MUTED, spaceAfter=6)
say = ParagraphStyle("say", fontName="Helvetica", fontSize=19, leading=29, textColor=INK, spaceAfter=22)

# (time, what to do on screen, what to say). " / " marks a natural breath.
BLOCKS = [
    ("BLOCK A · Slides 1–2", [
        ("0:00 – 0:15", "SLIDE 1 · cover",
         "Hi, I'm Diego. / This is a dispute agent for LATAM Bank / that knows when <b>not</b> to act. / "
         "It handles card disputes end to end, / in Spanish and Portuguese, / and hands off to a person when "
         "policy says so."),
        ("0:15 – 0:35", "SLIDE 2 · why disputes",
         "We chose disputes from the data. / Complaint contacts are fixed on first contact only "
         "<b>44 percent</b> of the time, / against <b>92 percent</b> for transactional ones, / and unrecognized "
         "charges and undue fees are about <b>40 percent</b> of formal complaints. / Disputes also have a "
         "trustworthy ground truth: / every transaction ties cleanly to its product and customer, / so the agent "
         "can verify the exact charge."),
    ]),
    ("BLOCK B · Live demo", [
        ("0:35 – 0:45", "APP · empty chat (reload with Cmd+Shift+R first)",
         "Here it is, live. / The interface is in Spanish because that's what LATAM Bank customers speak; / the "
         "agent works in Spanish and Portuguese, and I'll show both. / On the right, demo scenarios with real "
         "sandbox customers; / on top, a progress bar that follows the agent's real stage; / and a live trace of "
         "every decision."),
        ("0:45 – 1:05", "CLICK  Cargo ambiguo: [&gt; Probar (PT)]  ·  then green  Enviar documento  ·  Enviar código",
         "A customer writes in Portuguese about a charge at Boutique Moda. / The router detects Portuguese and a "
         "dispute, / and before looking at any data, the agent verifies identity: / document, then a one-time "
         "code, / simulated here as an SMS."),
        ("1:05 – 1:20", "TYPE  o de 165   (or click option 1)",
         "There are two charges at that merchant, / so instead of guessing, it asks. / I answer in my own words, "
         "/ and the model maps that to the right option."),
        ("1:20 – 1:35", "TYPE  o que acontece se eu confirmar?",
         "Before any write, it asks for an explicit yes. / A question is not a yes: / it asks again, and writes "
         "nothing."),
        ("1:35 – 1:50", "CLICK  Sim, abrir contestação  ·  Sim, bloquear  ·  point at the bar, scroll the trace",
         "Now it files the dispute and offers to block the card. / In the trace you can see each tool call, / "
         "the read-back that verifies the write, / the policy rule, / and the cost of the turn: fractions of a "
         "cent."),
        ("1:50 – 2:05", "CLICK  [&gt; Testar outro cenário], then Monto alto: [&gt; Probar (ES)]  ·  document  ·  code",
         "And when it should <b>not</b> act: / a large withdrawal. / Policy says a person must review it, / so "
         "it hands off with a reference, / and the advisor receives everything already verified."),
    ]),
    ("BLOCK C · Slides 3–6", [
        ("2:05 – 2:30", "SLIDE 3 · architecture",
         "The core decision: / <b>the model understands, code decides.</b> / A learned router and an LLM read "
         "the message; / the LLM never sees the document or the code. / A state machine, a versioned policy, / "
         "and customer-scoped bank tools decide and act. / When the router and the model disagree, / the agent "
         "asks the customer instead of guessing."),
        ("2:30 – 2:45", "SLIDE 4 · router  &gt;  SLIDE 5 · results",
         "The router was chosen by a rule fixed before the sealed test set was opened, / and it beats keywords by "
         "more than <b>20 points</b>. / End to end, on <b>230</b> held-out conversations on fresh records, / the "
         "hybrid safely resolves <b>98 percent</b> of disputes, / against <b>86 and a half</b> for rules, / makes "
         "every required handoff, / and has <b>one</b> unsafe outcome, which we report."),
        ("2:45 – 3:00", "SLIDE 6 · production",
         "It runs today with tracing, retries, a spending cap, and a safe fallback, / and we list what is missing "
         "to make it real. / Build something that works, / prove it works, / and know when <b>not</b> to act. / "
         "Thank you."),
    ]),
]


def story():
    s = [Paragraph("LATAM Bank Dispute Agent · video pitch", title),
         Paragraph("Read the large text. Grey lines are what to do on screen; '/' marks a breath. "
                   "Target 3:00, hard max 3:30. Record block by block and join them.", small),
         Spacer(1, 18), Paragraph("Before recording", block)]
    for item in ["Open <b>latam-dispute-agent.onrender.com</b> 2 minutes early; reload until the 5 scenarios appear.",
                 "Hard reload once: <b>Cmd + Shift + R</b> (you must see '&gt; Probar' buttons).",
                 "One browser tab, zoom 110–125%, Do Not Disturb on.",
                 "Slides open in Present mode in another window (Cmd-Tab to switch).",
                 "Use the scenario buttons; never type document numbers (customers rotate between takes).",
                 "If a reply takes 5–9 s, keep talking about the trace, or cut the pause when editing."]:
        s.append(Paragraph(item, check, bulletText="[  ]"))
    s.append(PageBreak())
    for b, (name, rows) in enumerate(BLOCKS):
        if b:
            s.append(PageBreak())
        s.append(Paragraph(name, block))
        for when, do, text in rows:
            s.append(KeepTogether([Paragraph(f"{when}   ·   {do}", cue), Paragraph(text, say)]))
    return s


def footer(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"page {doc.page}")


SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                  topMargin=0.75 * inch, bottomMargin=0.8 * inch, title="Video pitch script",
                  author="Diego").build(story(), onFirstPage=footer, onLaterPages=footer)
print(OUT)
