#!/usr/bin/env python3
"""Build the final presentation deck for the Ames residential valuation project.

Every number in this deck comes from docs/results_fact_sheet.md or one of the
supporting documents (business understanding, model card, monitoring).  The deck is
built from scratch with python-pptx, 16:9, and uses the repository figures in
artifacts/figures plus the Streamlit screenshot in artifacts/reports.

Run with:  uv run python slides/build_deck.py
Output:    slides/residential_valuation_deck.pptx
Checks:    uv run python slides/build_deck.py --check   (renders nothing, prints a
           structural and text-fit report for the saved deck)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "figures"
REPORTS = ROOT / "artifacts" / "reports"
ASSETS = Path(__file__).resolve().parent / "assets"
OUT = Path(__file__).resolve().parent / "residential_valuation_deck.pptx"

# --------------------------------------------------------------------------- design
SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.62
CONTENT_W = SLIDE_W - 2 * MARGIN  # 12.093
KICKER_Y = 0.30
TITLE_Y = 0.58
TITLE_H = 0.94
RULE_Y = 1.62
BODY_TOP = 1.90
BODY_BOTTOM = 6.58
FOOT_RULE_Y = 6.86
FOOTNOTE_Y = 6.95

SERIF = "Palatino Linotype"

NAVY = RGBColor(0x1B, 0x31, 0x45)
INK = RGBColor(0x25, 0x2B, 0x31)
TEAL = RGBColor(0x2E, 0x5F, 0x5B)
COPPER = RGBColor(0xA2, 0x61, 0x3C)
GRAY = RGBColor(0x5B, 0x61, 0x67)
MUTED = RGBColor(0x83, 0x89, 0x8F)
RULE = RGBColor(0xCE, 0xC7, 0xB9)
PANEL = RGBColor(0xF7, 0xF4, 0xEC)
PANEL_COOL = RGBColor(0xEF, 0xF3, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

TOTAL_SLIDES = 14

# ------------------------------------------------------------------------ primitives
def _strip_style(shape):
    """Remove the theme shape style so explicit fills and hairlines are honoured."""
    style = shape._element.find(qn("p:style"))
    if style is not None:
        shape._element.remove(style)


def rect(slide, x, y, w, h, fill=None, line=None, line_w=0.75):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    _strip_style(shape)
    shape.shadow.inherit = False
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_w)
    shape.text_frame.word_wrap = True
    return shape


def hline(slide, x1, y, x2, color=RULE, width=0.75):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y), Inches(x2), Inches(y)
    )
    _strip_style(line)
    line.shadow.inherit = False
    line.line.color.rgb = color
    line.line.width = Pt(width)
    return line


def arrow(slide, x1, y1, x2, y2, color=COPPER, width=1.0):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    _strip_style(line)
    line.shadow.inherit = False
    line.line.color.rgb = color
    line.line.width = Pt(width)
    ln = line.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return line


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tf


def _style_run(run, size, color, bold=False, italic=False, spc=None):
    f = run.font
    f.name = SERIF
    f.size = Pt(size)
    f.color.rgb = color
    f.bold = bold
    f.italic = italic
    if spc is not None:
        run.font._element.set("spc", str(spc))


def _para(tf, first, runs, *, size=11.5, color=INK, bold=False, italic=False,
          align=PP_ALIGN.LEFT, space_before=0, space_after=6, line=1.12,
          bullet=None, bullet_color=COPPER, indent=0.22, spc=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.line_spacing = line
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if isinstance(runs, str):
        runs = [(runs, {})]
    if bullet:
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(indent))))
        pPr.set("indent", str(-Emu(Inches(indent))))
        buClr = pPr.makeelement(qn("a:buClr"), {})
        clr = buClr.makeelement(qn("a:srgbClr"), {"val": str(bullet_color)})
        buClr.append(clr)
        buSz = pPr.makeelement(qn("a:buSzPct"), {"val": "68000"})
        buFont = pPr.makeelement(qn("a:buFont"), {"typeface": SERIF})
        buChar = pPr.makeelement(qn("a:buChar"), {"char": bullet})
        for el in (buClr, buSz, buFont, buChar):
            pPr.append(el)
    for text, opts in runs:
        r = p.add_run()
        r.text = text
        _style_run(
            r,
            opts.get("size", size),
            opts.get("color", color),
            opts.get("bold", bold),
            opts.get("italic", italic),
            opts.get("spc", spc),
        )
    return p


def kicker(slide, text):
    tf = textbox(slide, MARGIN, KICKER_Y, CONTENT_W, 0.26)
    _para(tf, True, [(text.upper(), {})], size=9, color=TEAL, spc=170, space_after=0)


def title(slide, text, size=23):
    tf = textbox(slide, MARGIN, TITLE_Y, CONTENT_W, TITLE_H)
    _para(tf, True, [(text, {})], size=size, color=NAVY, bold=True, line=1.04,
          space_after=0)


def header(slide, kicker_text, title_text, title_size=23):
    kicker(slide, kicker_text)
    title(slide, title_text, size=title_size)
    hline(slide, MARGIN, RULE_Y, MARGIN + CONTENT_W)


def footer(slide, number, source):
    hline(slide, MARGIN, FOOT_RULE_Y, MARGIN + CONTENT_W)
    tf = textbox(slide, MARGIN, FOOTNOTE_Y, CONTENT_W - 1.2, 0.28)
    _para(tf, True, [(source, {})], size=8.5, color=MUTED, italic=True, space_after=0)
    num = textbox(slide, MARGIN + CONTENT_W - 1.15, FOOTNOTE_Y, 1.15, 0.28)
    _para(num, True, [(f"{number:02d} / {TOTAL_SLIDES:02d}", {})], size=8.5,
          color=MUTED, align=PP_ALIGN.RIGHT, space_after=0)


def implication(slide, text, y=6.02):
    """A single editorial takeaway line above the source footnote."""
    hline(slide, MARGIN, y - 0.10, MARGIN + CONTENT_W, color=RULE, width=0.5)
    tf = textbox(slide, MARGIN, y, CONTENT_W, 0.40)
    _para(tf, True, [
        ("IMPLICATION", {"size": 8, "color": COPPER, "spc": 150}),
        ("     " + text[0].upper() + text[1:],
         {"size": 11, "color": NAVY, "italic": True}),
    ], space_after=0)


def bullets(slide, x, y, w, h, items, *, size=11.5, color=INK, space=8, line=1.16):
    """items: list of str or (lead, detail) tuples."""
    tf = textbox(slide, x, y, w, h)
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            lead, detail = item
            runs = [
                (lead + "  ", {"bold": True, "color": NAVY}),
                (detail, {"color": color}),
            ]
        else:
            runs = [(item, {"color": color})]
        _para(tf, i == 0, runs, size=size, space_after=space, line=line,
              bullet="▪", bullet_color=COPPER)
    return tf


def caption(slide, x, y, w, text, h=0.34, size=8.5):
    tf = textbox(slide, x, y, w, h)
    _para(tf, True, [(text, {})], size=size, color=MUTED, italic=True, line=1.1,
          space_after=0)


def figure(slide, path, x, y, box_w, box_h, frame=True, align="center"):
    """Place a picture inside the box preserving its aspect ratio."""
    img_w, img_h = Image.open(path).size
    scale = min(box_w / img_w, box_h / img_h)
    w, h = img_w * scale, img_h * scale
    if align == "center":
        px = x + (box_w - w) / 2
    else:
        px = x
    py = y
    if frame:
        rect(slide, px - 0.05, py - 0.05, w + 0.10, h + 0.10, fill=None, line=RULE,
             line_w=0.6)
    slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(w), Inches(h))
    return px, py, w, h


def _cell_borders(cell, bottom=RULE, width_emu=9525):
    """Explicit hairline rules: no left/right/top edges, one warm bottom rule."""
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        for el in tcPr.findall(qn(tag)):
            tcPr.remove(el)
    specs = [("a:lnB", bottom), ("a:lnT", None), ("a:lnR", None), ("a:lnL", None)]
    for tag, color in specs:
        ln = tcPr.makeelement(qn(tag), {"w": str(width_emu), "cap": "flat",
                                        "cmpd": "sng", "algn": "ctr"})
        if color is None:
            ln.append(ln.makeelement(qn("a:noFill"), {}))
        else:
            fill = ln.makeelement(qn("a:solidFill"), {})
            clr = fill.makeelement(qn("a:srgbClr"), {"val": str(color)})
            fill.append(clr)
            ln.append(fill)
        tcPr.insert(0, ln)


def table(slide, x, y, col_widths, rows, *, header_h=0.44, row_h=0.52,
          size=10.5, header_size=9.5, first_col_bold=False, aligns=None):
    """rows[0] is the header.  Thin-rule, navy-header table with explicit fills."""
    n_rows, n_cols = len(rows), len(col_widths)
    width = sum(col_widths)
    height = header_h + row_h * (n_rows - 1)
    gframe = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y),
                                    Inches(width), Inches(height))
    tbl = gframe.table
    tbl.first_row = False
    tbl.horz_banding = False
    tbl.vert_banding = False
    tblPr = tbl._tbl.tblPr
    for style in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(style)
    # "No Style, No Grid" so no inherited borders or fills survive
    style_el = tblPr.makeelement(qn("a:tableStyleId"), {})
    style_el.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    tblPr.append(style_el)

    for i, cw in enumerate(col_widths):
        tbl.columns[i].width = Inches(cw)
    tbl.rows[0].height = Inches(header_h)
    for r in range(1, n_rows):
        tbl.rows[r].height = Inches(row_h)

    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.09)
            cell.margin_right = Inches(0.09)
            cell.margin_top = Inches(0.045)
            cell.margin_bottom = Inches(0.045)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r == 0 else WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if aligns is None else aligns[c]
            p.line_spacing = 1.02
            p.space_before = Pt(0)
            p.space_after = Pt(0)
            run = p.add_run()
            run.text = value
            if r == 0:
                _style_run(run, header_size, WHITE, bold=True)
            else:
                _style_run(run, size, NAVY if (c == 0 and first_col_bold) else INK,
                           bold=(c == 0 and first_col_bold))
            _cell_borders(cell)
    return gframe


def node(slide, x, y, w, h, heading, body, accent=TEAL, fill=PANEL):
    box = rect(slide, x, y, w, h, fill=fill, line=RULE, line_w=0.75)
    box.text_frame.margin_left = Inches(0.18)
    box.text_frame.margin_right = Inches(0.12)
    box.text_frame.margin_top = Inches(0.06)
    box.text_frame.margin_bottom = Inches(0.05)
    box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    rect(slide, x, y, 0.045, h, fill=accent, line=None)
    tf = box.text_frame
    _para(tf, True, [(heading, {})], size=10.5, color=NAVY, bold=True, line=1.0,
          space_after=2)
    _para(tf, False, [(body, {})], size=9, color=GRAY, line=1.08, space_after=0)
    return box


# --------------------------------------------------------------------------- slides
def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, SLIDE_W, 0.10, fill=NAVY, line=None)

    tf = textbox(s, MARGIN, 0.86, CONTENT_W, 0.28)
    _para(tf, True, [("Business Analytics  ·  Group Project  ·  Final Presentation", {})],
          size=9.5, color=TEAL, spc=150, space_after=0)

    tf = textbox(s, MARGIN, 1.26, CONTENT_W - 0.6, 1.55)
    _para(tf, True, [("House Price Prediction for a", {})], size=40, color=NAVY,
          bold=True, line=1.02, space_after=0)
    _para(tf, False, [("Real Estate Listing Platform", {})], size=40, color=NAVY,
          bold=True, line=1.02, space_after=0)

    tf = textbox(s, MARGIN, 2.92, CONTENT_W - 1.2, 0.42)
    _para(tf, True, [("A calibrated valuation system for residential listings in Ames, Iowa",
                      {})], size=17, color=GRAY, italic=True, space_after=0)

    rect(s, MARGIN, 3.52, 1.9, 0.035, fill=COPPER, line=None)

    meta = [
        ("COURSE", "Business Analytics"),
        ("PRESENTATION", "November 2026"),
        ("SCOPE", "Modules 1–8: data to monitoring"),
    ]
    for i, (label, value) in enumerate(meta):
        x = MARGIN + i * 4.05
        tf = textbox(s, x, 3.78, 3.8, 0.24)
        _para(tf, True, [(label, {})], size=8.5, color=MUTED, spc=150, space_after=0)
        tf = textbox(s, x, 4.02, 3.8, 0.30)
        _para(tf, True, [(value, {})], size=11.5, color=NAVY, space_after=0)

    hline(s, MARGIN, 4.58, MARGIN + CONTENT_W)

    tf = textbox(s, MARGIN, 4.78, CONTENT_W, 0.24)
    _para(tf, True, [("PROJECT TEAM", {})], size=8.5, color=COPPER, spc=150,
          space_after=0)

    roster = [
        ("Data Engineer", "[Member 1 Name] ([Student ID 1])"),
        ("Data Analyst", "[Member 2 Name] ([Student ID 2])"),
        ("ML Modeling Engineer", "[Member 3 Name] ([Student ID 3])"),
        ("ML Evaluation Engineer", "[Member 4 Name] ([Student ID 4])"),
        ("MLOps Lead", "[Member 5 Name] ([Student ID 5])"),
    ]
    tf = textbox(s, MARGIN, 5.10, CONTENT_W, 1.50)
    for i, (role, person) in enumerate(roster):
        _para(tf, i == 0,
              [(role, {"bold": True, "color": NAVY}),
               ("  —  ", {"color": MUTED}),
               (person, {"color": INK})],
              size=11.5, space_after=5, line=1.0)

    footer(s, 1, "Base data: Kaggle House Prices – Advanced Regression Techniques, "
                 "1,460 labelled transactions (Ames, Iowa, 2006–2010) plus 21 documented "
                 "synthetic contextual fields.")
    return s


def slide_business(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Business question and KPIs",
           "Accuracy, calibration and cost are three separate targets, and all three were met")

    rows = [
        ["Indicator", "Target", "Achieved (out-of-fold)"],
        ["RMSE", "at most 25,000 USD", "20,019 USD"],
        ["MAE", "at most 17,000 USD", "13,537 USD"],
        ["MAPE", "at most 10%", "7.98%"],
        ["R-squared", "at least 0.90", "0.9365"],
        ["Within ±15,000 USD", "at least 65%", "69.7%"],
        ["80% / 90% coverage", "0.75–0.85 / 0.85–0.95", "0.801 / 0.901"],
    ]
    table(s, MARGIN, BODY_TOP, [2.05, 2.10, 2.15], rows, header_h=0.50, row_h=0.58,
          size=11, header_size=10, first_col_bold=True)

    bullets(s, 7.35, 1.96, 5.35, 4.6, [
        ("Cost of error.",
         "Over-valuation is weighted 1.5 against 1.0, giving a mean asymmetric cost of "
         "16,708 USD per listing and a median of 11,358 USD."),
        ("Interval, not a number.",
         "Agents receive an 80% and a 90% band with every estimate; the platform needs "
         "coverage it can plan around, not a single figure."),
        ("Two-sided coverage.",
         "An interval that is too wide is as useless as one that is too narrow, so the "
         "targets carry both a floor and a ceiling."),
        ("Escalation.",
         "Estimates whose 90% interval exceeds 30% of the point estimate count as low "
         "confidence and are routed to a valuer before publication."),
    ], size=11.5, space=11)

    footer(s, 2, "Source: out-of-fold evaluation of 1,458 transactions — results fact "
                 "sheet, sections 4 and 5; business understanding, sections 3 and 4.")
    return s


def slide_architecture(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Data architecture",
           "A Kaggle core plus 21 simulated context fields, with leakage blocked in code")

    boxes = [
        (MARGIN, 2.95, "Kaggle base",
         "1,460 labelled and 1,459 unlabelled sales, 79 attributes, 2006–2010."),
        (3.86, 2.95, "Synthetic context",
         "21 fields in five families, generated once with seed 42 and joined to every "
         "record."),
        (7.10, 2.55, "Cleaned analytical table",
         "1,458 transactions after 7,829 missing cells and two implausible records were "
         "handled."),
        (9.94, 2.74, "Model inputs",
         "102 features; sale condition, sale type, days on market and listing date "
         "excluded."),
    ]
    for x, w, head, body in boxes:
        node(s, x, 1.94, w, 1.52, head, body)

    arrow(s, 3.60, 2.70, 3.83, 2.70)
    arrow(s, 6.84, 2.70, 7.07, 2.70)
    arrow(s, 9.68, 2.70, 9.91, 2.70)

    rect(s, MARGIN, 3.76, CONTENT_W, 0.86, fill=PANEL, line=None)
    rect(s, MARGIN, 3.76, 0.045, 0.86, fill=COPPER, line=None)
    tf = textbox(s, MARGIN + 0.24, 3.87, CONTENT_W - 0.5, 0.22)
    _para(tf, True, [("EXCLUDED FROM THE MODEL INPUTS", {})], size=8.5, color=COPPER,
          spc=140, space_after=0)
    tf = textbox(s, MARGIN + 0.24, 4.14, CONTENT_W - 0.5, 0.40)
    _para(tf, True, [("SalePrice  ·  log_price  ·  price_per_sqm  ·  sale_condition  ·  "
                      "sale_type  ·  days_on_market  ·  listing_date  ·  yr_sold  ·  "
                      "mo_sold  ·  id", {})],
          size=9.5, color=INK, italic=True, line=1.1, space_after=0)

    bullets(s, MARGIN, 4.94, CONTENT_W, 1.55, [
        ("Leakage is enforced, not assumed.",
         "These fields are dropped in code and covered by a test, because none of them is "
         "observable when a listing is published."),
        ("Context is documented and reproducible.",
         "Every synthetic field records type, unit, valid range, generation logic and "
         "assumption; the generator never reads SalePrice and is seeded with 42."),
    ], size=11.5, space=7)

    footer(s, 3, "Source: data dictionary, sections 4 and 7; business understanding, "
                 "sections 5 and 7; results fact sheet, section 1.")
    return s


def slide_synthetic(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Synthetic context",
           "Context fields follow documented rules, then are checked against observed prices")

    rows = [
        ["Family", "Fields", "Generation rule and realism check"],
        ["Location and amenity access", "9",
         "Distances tighten with tier; the amenity score averages 85.5 in tier-5 against "
         "43.5 in tier-1 neighbourhoods."],
        ["Neighbourhood quality", "3",
         "School rating, walkability and crime index are drawn from the tier prior; the crime "
         "index correlates at −0.55 with sale price."],
        ["Renovation history", "2",
         "Flag and scope follow the remodelling lag; age at sale carries 3.41% of the global "
         "SHAP explanation mass."],
        ["Listing duration", "2",
         "Days on market load on tier, season, mortgage rate and supply; median 31 days, "
         "Spearman −0.435 with price."],
        ["Macro-financial", "5",
         "Anchored monthly path for the 2006–2011 cycle; mortgage rate 6.4% to 4.9%, house "
         "price index 104 to 96."],
    ]
    table(s, MARGIN, BODY_TOP, [2.35, 0.80, 8.94], rows, header_h=0.46, row_h=0.52,
          size=10.5, header_size=9.5, first_col_bold=True,
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.LEFT])

    bullets(s, MARGIN, 5.46, CONTENT_W, 1.2, [
        ("Audited, not assumed.",
         "Realism is tested by association with observed prices: 0.652 Spearman for the "
         "amenity score, −0.55 Pearson for crime and −0.435 for days on market."),
        ("Reproducible.",
         "Generation is deterministic for a given seed and input order, so the data "
         "dictionary, the table and the report can be regenerated together."),
    ], size=11, space=6)

    footer(s, 4, "Source: docs/data_dictionary.md section 4; "
                 "docs/business_understanding.md section 5; results fact sheet, "
                 "sections 1 and 2.")
    return s


def slide_eda(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Exploratory analysis",
           "Price is log-normal in practice, and area and quality carry the raw signal")

    figure(s, FIG / "fig01_price_distribution.png", MARGIN, BODY_TOP, 8.20, 3.38)
    caption(s, MARGIN, 5.36, 8.20,
            "Figure 1.  1,460 labelled sales: median 163,000 USD, mean 180,933 USD; "
            "skewness 1.881 falls to 0.122 on the log scale.")

    bullets(s, 8.90, 1.94, 3.80, 4.6, [
        ("Log target.",
         "Skewness of 1.881 becomes 0.122 after ln(1 + price), so the model is fitted on "
         "the log scale."),
        ("Positive drivers.",
         "Total floor area 0.83, overall quality 0.80 and living area 0.74 lead the "
         "Pearson ranking."),
        ("Negative drivers.",
         "Crime index −0.55, age at sale −0.52 and transit distance −0.37 pull the other "
         "way."),
        ("Outliers.",
         "61 sales (4.2%) fall outside the interquartile range; they are flagged rather "
         "than dropped, because the log target keeps them informative."),
    ], size=11, space=9)

    implication(s, "every leading driver is observable at listing time, so the model can "
                   "be explained field by field to a listing agent.")

    footer(s, 5, "Source: fig01; correlation values from fig03, computed on the 1,460 "
                 "labelled transactions (results fact sheet, sections 1 and 2).")
    return s


def slide_segments(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Market segmentation",
           "Location buys a 2.8× spread in median price, and it is visible before any model")

    figure(s, FIG / "fig05_amenity_price.png", MARGIN, BODY_TOP, 8.00, 3.41)
    caption(s, MARGIN, 5.39, 8.00,
            "Figure 2.  1,458 cleaned sales by amenity score and sub-market; premium areas "
            "combine high amenity access with the widest price range.")

    bullets(s, 8.85, 1.94, 3.85, 4.6, [
        ("Tier medians.",
         "109,000 / 125,500 / 148,000 / 196,750 / 306,000 USD from tier 1 to tier 5, a "
         "2.8× spread across the city."),
        ("Amenity gradient.",
         "Tier-5 neighbourhoods average 85.5 on the amenity score against 43.5 in tier-1 "
         "neighbourhoods (Spearman 0.652 with price)."),
        ("Market confirmation.",
         "Days on market fall as price rises (Spearman −0.435, median 31 days), so the "
         "spread is priced rather than incidental."),
        ("Band mix.",
         "365 entry, 658 mid, 320 upper and 115 luxury sales: the mid band dominates, "
         "while the luxury band is thin and hardest to price."),
    ], size=11, space=9)

    implication(s, "neighbourhood tier and amenity access are both known at listing "
                   "time, so the segmentation is actionable in the console.")

    footer(s, 6, "Source: fig05; tier medians, correlations and band counts from results "
                 "fact sheet, sections 2 and 7; 1,458 cleaned transactions.")
    return s


def slide_cleaning(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Cleaning and feature engineering",
           "Cleaning dropped two sales and VIF pruning eight redundant measures")

    rows = [
        ["Stage", "Before", "After or decision"],
        ["Missing cells (train)", "7,829",
         "Structural absence mapped to “None”; lot frontage to the neighbourhood median"],
        ["Duplicate records", "checked", "None found across the 1,460 labelled records"],
        ["Implausible sales", "2 records",
         "Above 4,000 sqft and below 300,000 USD; removed, leaving 1,458 transactions"],
        ["Price outliers", "61 (4.2%)",
         "Flagged but retained; the log target absorbs the remaining skew"],
        ["Numeric features", "55",
         "47 retained by VIF pruning, with a maximum final VIF of 8.95"],
        ["Model inputs", "79 attributes",
         "102 features after 18 engineered variables are added"],
    ]
    table(s, MARGIN, BODY_TOP, [2.15, 1.55, 8.39], rows, header_h=0.46, row_h=0.45,
          size=10.5, header_size=9.5, first_col_bold=True)

    bullets(s, MARGIN, 5.30, 3.75, 1.40, [
        ("Collinear pairs.",
         "Total floor area and living area correlate at 0.87, garage area and capacity at "
         "0.88; composites replace the raw pairs."),
    ], size=11, space=6)
    bullets(s, 4.62, 5.30, 3.75, 1.40, [
        ("VIF evidence.",
         "Pruning keeps 47 of 55 numeric features and holds the maximum VIF at 8.95, so "
         "the linear coefficients stay stable."),
    ], size=11, space=6)
    bullets(s, 8.62, 5.30, 3.75, 1.40, [
        ("Target-safe features.",
         "price_per_sqm is reporting-only and excluded from the inputs because it is a "
         "direct function of the sale price."),
    ], size=11, space=6)

    footer(s, 7, "Source: results fact sheet, sections 1 and 2; "
                 "docs/data_dictionary.md sections 5–7; artifacts/metrics/vif.csv.")
    return s


def slide_benchmark(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Model benchmark",
           "Under one five-fold protocol, regularised linear models beat every tree ensemble")

    figure(s, FIG / "fig15_model_comparison.png", MARGIN, BODY_TOP, 8.30, 3.66)
    caption(s, MARGIN, 5.64, 8.30,
            "Figure 3.  Eight regression families under identical five-fold "
            "cross-validation; elastic net highlighted.")

    bullets(s, 9.12, 1.94, 3.58, 4.7, [
        ("Spread.",
         "Elastic net 20,019 USD RMSE against 22,471 for XGBoost, 22,518 for LightGBM and "
         "25,725 for random forest."),
        ("Consistency.",
         "The ordering holds across RMSE, MAE and MAPE, so the selection rule does not "
         "depend on one metric."),
        ("Why linear.",
         "The model card attributes the result to a small, well-measured tabular sample, "
         "where regularisation controls variance better than tree depth."),
        ("Temporal holdout.",
         "Fitted on 2006–2009 and tested on 2010: RMSE 22,269 USD, MAPE 8.71% and "
         "R-squared 0.923 on 175 sales."),
    ], size=11, space=9)

    footer(s, 8, "Source: fig15; artifacts/metrics/model_comparison.csv; model card, "
                 "sections 1 and 4 (results fact sheet, sections 3 and 4).")
    return s


def slide_performance(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Model performance and residuals",
           "The selected model reaches R² 0.9365, with error widening at the top")

    figure(s, FIG / "fig16_predicted_vs_actual.png", MARGIN, BODY_TOP, 5.20, 4.05)
    caption(s, MARGIN, 6.01, 5.20,
            "Figure 4.  Out-of-fold predictions against observed prices on 1,458 sales.")

    figure(s, FIG / "fig17_residual_diagnostics.png", 6.18, BODY_TOP, 6.50, 2.37)
    caption(s, 6.18, 4.33, 6.50,
            "Figure 5.  Log residuals against fitted value, their distribution and a normal "
            "Q–Q plot.")

    bullets(s, 6.18, 4.56, 6.50, 2.10, [
        ("Central accuracy.",
         "MAE 13,537 USD and median absolute error 9,356 USD; 69.7% of estimates fall "
         "within ±15,000 USD."),
        ("Residual shape.",
         "Log residuals are left-skewed (−1.14) and heteroscedastic (Breusch–Pagan "
         "p < 0.001), so Gaussian intervals would mislead."),
        ("Luxury tail.",
         "The 115 luxury sales carry MAE 31,238 USD and a −3.92% bias, and error grows "
         "with price."),
    ], size=11, space=5)

    footer(s, 9, "Source: fig16 and fig17; results fact sheet, sections 4 and 6; model "
                 "card, sections 4 and 5.")
    return s


def slide_intervals(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Calibrated uncertainty",
           "Intervals hold overall at 80.1% and 90.1%, but the entry band under-covers")

    figure(s, FIG / "fig20_interval_coverage.png", MARGIN, BODY_TOP, 7.90, 3.36)
    caption(s, MARGIN, 5.34, 7.90,
            "Figure 6.  Empirical coverage and mean width of the 80% and 90% conformal "
            "intervals, overall and by price band.")

    bullets(s, 8.76, 1.94, 3.94, 4.6, [
        ("Overall calibration.",
         "Coverage is 0.801 at the 80% level and 0.901 at the 90% level, within a "
         "thousandth of the nominal target."),
        ("Segment gaps.",
         "At 90%, coverage runs 83.3% in the entry band, 94.1% mid, 90.9% upper and 87.0% "
         "luxury; entry misses the 0.85 floor."),
        ("Width.",
         "The 90% interval averages 33.3% of price, from 31.8% in luxury to 34.7% in "
         "entry, so narrowness is not bought with accuracy."),
        ("Recalibration path.",
         "Recalibrating the conformal quantiles is the first action in the runbook; a refit "
         "follows only if the erosion persists."),
    ], size=11, space=9)

    implication(s, "both bands ship with every estimate, and the quantiles are refreshed "
                   "before any refit is considered.")

    footer(s, 10, "Source: fig20; artifacts/metrics/interval_coverage.csv; model card, "
                  "sections 4 and 5; monitoring runbook, section 4.")
    return s


def slide_drivers(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Interpretation",
           "Structure and quality carry 43% of the explanation mass; location 3.2%")

    figure(s, FIG / "fig18_global_drivers.png", MARGIN, BODY_TOP, 6.60, 4.36)
    caption(s, MARGIN, 6.30, 6.60,
            "Figure 7.  Mean absolute SHAP share by attribute, coloured by feature family.")

    bullets(s, 7.48, 1.94, 5.22, 4.6, [
        ("Leading attributes.",
         "Rooms above grade 8.73%, total floor area 6.51% and the quality–condition index "
         "6.31% dominate the global ranking."),
        ("Family shares.",
         "Structural 43.35%, categorical 27.82%, composites 15.87%, age 6.21%, location "
         "3.20%, amenities 2.40% and macro-financial 1.14%."),
        ("Location and age.",
         "Latitude enters at 3.05% and age at sale at 3.41%, both consistent with the "
         "negative age–price correlation seen in the EDA."),
        ("Macro context.",
         "Macro-financial fields carry 1.14% of the explanation mass, so their influence "
         "arrives as market drift rather than as a per-listing driver."),
        ("Reading caution.",
         "The data are observational and the context is simulated, so no coefficient in "
         "this model carries a causal interpretation."),
    ], size=11, space=14)

    footer(s, 11, "Source: fig18; artifacts/metrics/shap_importance.csv (results fact "
                  "sheet, section 8); model card, section 6; monitoring, section 4.")
    return s


def slide_product(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Product and deployment",
           "The model is served through an API and a console, verified locally")

    nodes = [
        ("Streamlit console",
         "Valuation, model performance, monitoring and data-and-methodology views."),
        ("FastAPI service",
         "/predict returns the point estimate, both intervals and the local drivers."),
        ("Elastic net v1.0.0",
         "102 inputs on the log price, with conformal quantiles from out-of-fold residuals."),
        ("Monitoring artefacts",
         "PSI and KS drift, rolling accuracy, bias and interval coverage."),
    ]
    y = 1.94
    for i, (head, body) in enumerate(nodes):
        node(s, MARGIN, y, 5.15, 0.66, head, body)
        if i < len(nodes) - 1:
            arrow(s, MARGIN + 2.57, y + 0.67, MARGIN + 2.57, y + 0.79, width=0.9)
        y += 0.79

    bullets(s, MARGIN, 5.12, 5.15, 1.55, [
        ("A single contract.",
         "The console calls the API over VALUATION_API_URL and falls back to an in-process "
         "pipeline when the service is unreachable."),
        ("Explainable by default.",
         "Each response carries per-attribute contributions and a confidence score, so "
         "low-confidence listings can be routed to a valuer."),
    ], size=11, space=6)

    figure(s, ASSETS / "ui_result_console.png", 6.05, 1.92, 6.63, 2.95)
    caption(s, 6.05, 4.97, 6.63,
            "Figure 8.  Console response for a College Creek listing: point estimate, 80% and "
            "90% bands, confidence score and local drivers.", h=0.42)

    rect(s, 6.05, 5.50, 6.63, 0.98, fill=PANEL, line=None)
    rect(s, 6.05, 5.50, 0.045, 0.98, fill=COPPER, line=None)
    tf = textbox(s, 6.24, 5.60, 6.28, 0.80)
    _para(tf, True, [("VERIFIED LOCALLY — NO PUBLIC URL IS CLAIMED", {})], size=8.5,
          color=COPPER, spc=140, space_after=3)
    _para(tf, False, [("API health and prediction, the console valuation page and the "
                       "monitoring report were exercised locally; Render, Docker and "
                       "Hugging Face Spaces configurations are present in deploy/ and "
                       "render.yaml.", {})],
          size=9.5, color=INK, line=1.1, space_after=0)

    footer(s, 12, "Source: artifacts/reports/ui_result.png; docs/deployment.md; results "
                  "fact sheet, section 10.")
    return s


def slide_monitoring(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Monitoring",
           "Macro drift was expected; the coverage erosion it caused was not")

    figure(s, FIG / "fig22_drift_heatmap.png", MARGIN, BODY_TOP, 7.20, 3.56)
    caption(s, MARGIN, 5.52, 7.20,
            "Figure 9.  Population stability index by monitoring window; the macro block "
            "breaches 0.20 in every window, while structural features stay stable.")

    bullets(s, 8.12, 1.94, 4.58, 2.85, [
        ("Drift was concentrated.",
         "Five to seven features exceed PSI 0.20 in every window: mortgage rate, house "
         "price index, unemployment and months of supply."),
        ("Accuracy held.",
         "The worst window is 1.12× the 20,888 USD reference RMSE, below the 1.2× critical "
         "threshold, so the accuracy rule never fired."),
        ("Calibration did not.",
         "90% coverage falls from 89.1% to 79.7%, under the 0.85 floor twice, and "
         "prediction PSI rises from 0.066 to 0.174."),
    ], size=11, space=9)

    rect(s, 8.12, 4.72, 4.58, 1.06, fill=PANEL, line=None)
    rect(s, 8.12, 4.72, 0.045, 1.06, fill=COPPER, line=None)
    tf = textbox(s, 8.31, 4.82, 4.25, 0.88)
    _para(tf, True, [("POLICY OUTCOME", {})], size=8.5, color=COPPER, spc=140,
          space_after=3)
    _para(tf, False, [("R7, sustained coverage erosion, is critical: retrain now, "
                       "recalibrating the conformal quantiles first and then refitting on "
                       "the most recent 24 months of labelled data.", {})],
          size=9.5, color=INK, line=1.12, space_after=0)

    footer(s, 13, "Source: fig22; docs/monitoring.md sections 3 and 4 — reference model on "
                  "945 sales (2006–2008), stream of 513 sales (2009–2010) in four windows.")
    return s


def slide_closing(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Recommendations and limitations",
           "Adopt the estimate with guardrails: three operating rules and two honest limits")

    tf = textbox(s, MARGIN, 1.92, CONTENT_W, 0.24)
    _para(tf, True, [("RECOMMENDATIONS", {})], size=8.5, color=COPPER, spc=150,
          space_after=0)
    bullets(s, MARGIN, 2.22, CONTENT_W, 1.55, [
        ("Publish the interval, not just the price.",
         "Agents get both bands with a confidence score; estimates whose 90% interval "
         "exceeds 30% of the point estimate are routed to a valuer."),
        ("Recalibrate monthly, retrain on trigger.",
         "Refresh the conformal quantiles every month and refit on the most recent 24 "
         "months when a critical rule fires."),
        ("Replace the simulation before reading drivers.",
         "Amenity, renovation and macro fields must come from contracted feeds before "
         "their coefficients are treated as measured effects."),
    ], size=11.5, space=10)

    hline(s, MARGIN, 3.92, MARGIN + CONTENT_W, color=RULE, width=0.5)

    tf = textbox(s, MARGIN, 4.10, CONTENT_W, 0.24)
    _para(tf, True, [("DOCUMENTED LIMITATIONS", {})], size=8.5, color=COPPER, spc=150,
          space_after=0)
    bullets(s, MARGIN, 4.40, CONTENT_W, 1.30, [
        ("Evidential limits.",
         "1,458 transactions and simulated context mean the coefficients describe the "
         "simulation, not causal effects, and luxury sales carry MAE 31,238 USD."),
        ("Regime limits.",
         "No high-interest-rate regime has been observed, and one global calibration "
         "leaves the entry band under-covered at 83.3%."),
    ], size=11.5, space=10)

    rect(s, MARGIN, 5.66, CONTENT_W, 0.82, fill=PANEL_COOL, line=None)
    rect(s, MARGIN, 5.66, 0.045, 0.82, fill=TEAL, line=None)
    tf = textbox(s, MARGIN + 0.26, 5.86, CONTENT_W - 0.5, 0.52)
    _para(tf, True, [("CONCLUSION", {})], size=8.5, color=TEAL, spc=150, space_after=2)
    _para(tf, False, [("Every statistical KPI was met on 1,458 transactions, the product is "
                       "verified locally, and the monitoring policy fires before the "
                       "estimate stops being defensible.", {})],
          size=12, color=NAVY, italic=True, line=1.1, space_after=0)

    footer(s, 14, "Source: results fact sheet, section 4; model card, sections 8–10; "
                  "monitoring runbook, section 4.")
    return s


# ------------------------------------------------------------------- screenshot asset
def prepare_assets():
    """Crop the console response out of the screenshot into slides/assets/."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    src = REPORTS / "ui_result.png"
    dst = ASSETS / "ui_result_console.png"
    with Image.open(src) as im:
        crop = im.crop((360, 925, 1440, 1400)).convert("RGB")
        crop.save(dst, optimize=True)
    return dst


# --------------------------------------------------------------------- deck assembly
def build():
    prepare_assets()
    prs = Presentation()
    prs.slide_width = Emu(12192000)   # 13.333 in
    prs.slide_height = Emu(6858000)   # 7.5 in

    slide_title(prs)
    slide_business(prs)
    slide_architecture(prs)
    slide_synthetic(prs)
    slide_eda(prs)
    slide_segments(prs)
    slide_cleaning(prs)
    slide_benchmark(prs)
    slide_performance(prs)
    slide_intervals(prs)
    slide_drivers(prs)
    slide_product(prs)
    slide_monitoring(prs)
    slide_closing(prs)

    prs.core_properties.title = "House Price Prediction for a Real Estate Listing Platform"
    prs.core_properties.subject = ("Ames residential valuation: calibrated model, product, "
                                   "monitoring")
    prs.core_properties.author = "Business Analytics group project"
    prs.core_properties.comments = ("Built from docs/results_fact_sheet.md and repository "
                                    "artefacts; all figures are local files.")
    prs.save(OUT)
    return OUT


# ------------------------------------------------------------------------ validation
FONT_FILES = {
    (False, False): "/usr/share/fonts/gsfonts/P052-Roman.otf",
    (True, False): "/usr/share/fonts/gsfonts/P052-Bold.otf",
    (False, True): "/usr/share/fonts/gsfonts/P052-Italic.otf",
    (True, True): "/usr/share/fonts/gsfonts/P052-BoldItalic.otf",
}
SCALE = 4.0  # pixels per point when measuring


def _font(size_pt, bold, italic):
    path = FONT_FILES[(bold, italic)]
    return ImageFont.truetype(path, int(round(size_pt * SCALE)))


def _text_width_pt(text, size_pt, bold, italic):
    if not text:
        return 0.0
    return _font(size_pt, bold, italic).getlength(text) / SCALE


def _wrap_lines(runs, avail_pt):
    """Greedy wrap of styled runs; returns the number of lines and the max size."""
    words = []
    max_size = 0.0
    for text, size, bold, italic in runs:
        max_size = max(max_size, size)
        for i, chunk in enumerate(text.split(" ")):
            if chunk == "":
                continue
            words.append((chunk, size, bold, italic))
        if text.endswith(" "):
            words.append(("", size, bold, italic))
    if not words:
        return 1, max_size or 11.0
    space = _text_width_pt(" ", words[0][1], words[0][2], words[0][3])
    lines, current = 1, 0.0
    for chunk, size, bold, italic in words:
        if chunk == "":
            current += space
            continue
        w = _text_width_pt(chunk, size, bold, italic)
        if current > 0 and current + space + w > avail_pt:
            lines += 1
            current = w
        else:
            current += (space if current > 0 else 0) + w
    return lines, max_size


def _paragraph_runs(p):
    runs = []
    for r in p.runs:
        size = r.font.size.pt if r.font.size else 11.0
        runs.append((r.text, size, bool(r.font.bold), bool(r.font.italic)))
    return runs


def _estimate_tf_height(tf, width_in, default_size=11.5):
    total = 0.0
    for p in tf.paragraphs:
        pPr = p._p.find(qn("a:pPr"))
        indent = 0.0
        if pPr is not None and pPr.get("marL"):
            indent = int(pPr.get("marL")) / 914400.0
        avail = (width_in - indent) * 72.0
        runs = _paragraph_runs(p) or [("", default_size, False, False)]
        lines, max_size = _wrap_lines(runs, max(avail, 10))
        spacing = p.line_spacing if isinstance(p.line_spacing, float) else 1.0
        total += lines * max_size * 1.19 * spacing
        total += (p.space_before.pt if p.space_before else 0)
        total += (p.space_after.pt if p.space_after else 0)
    return total


def validate(path=OUT):
    prs = Presentation(str(path))
    problems = []
    print(f"slides: {len(prs.slides)}   size: "
          f"{prs.slide_width / 914400:.3f} x {prs.slide_height / 914400:.3f} in")
    if len(prs.slides) != TOTAL_SLIDES:
        problems.append(f"expected {TOTAL_SLIDES} slides, found {len(prs.slides)}")
    if abs(prs.slide_width / 914400 - 13.333) > 0.01 or \
       abs(prs.slide_height / 914400 - 7.5) > 0.01:
        problems.append("slide size is not 16:9 13.333 x 7.5 in")

    for idx, slide in enumerate(prs.slides, start=1):
        for sh in slide.shapes:
            left = sh.left / 914400
            top = sh.top / 914400
            width = sh.width / 914400
            height = sh.height / 914400
            if left < -0.02 or top < -0.02 or left + width > SLIDE_W + 0.02 \
               or top + height > SLIDE_H + 0.02:
                problems.append(
                    f"slide {idx}: {sh.shape_type} '{sh.name}' outside the slide "
                    f"({left:.2f},{top:.2f},{width:.2f},{height:.2f})")
            if sh.has_text_frame and sh.text_frame.text.strip():
                ml = sh.text_frame.margin_left / 914400
                mr = sh.text_frame.margin_right / 914400
                mt = sh.text_frame.margin_top / 914400
                mb = sh.text_frame.margin_bottom / 914400
                est = (_estimate_tf_height(sh.text_frame, width - ml - mr) / 72.0
                       + mt + mb)
                if est > height + 0.03:
                    problems.append(
                        f"slide {idx}: text may overflow '{sh.name}' "
                        f"(est {est:.2f} in vs box {height:.2f} in) "
                        f"'{sh.text_frame.text[:48]}'")
    if problems:
        print(f"checks: {len(problems)} issue(s)")
        for p in problems:
            print("  -", p)
    else:
        print("checks: no structural or text-fit problems found")
    return problems


if __name__ == "__main__":
    if "--check" in sys.argv:
        validate()
        sys.exit(0)
    out = build()
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.2f} MB)")
    validate()
    sys.exit(0)
