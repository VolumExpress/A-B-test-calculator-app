from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Mapping
import html

import pandas as pd
from jinja2 import Template
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


HTML_TEMPLATE = Template("""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>{{ title }}</title>
<style>
body{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;color:#1f2937;line-height:1.45}
h1{background:#8F1734;color:white;padding:18px 22px;border-radius:8px} h2{color:#243447;border-bottom:2px solid #dde3ea;padding-bottom:6px}
table{border-collapse:collapse;width:100%;margin:12px 0 24px;font-size:14px} th{background:#243447;color:#fff;text-align:left}
th,td{border:1px solid #cbd5e1;padding:7px;vertical-align:top}.note{background:#f3f5f7;padding:12px;border-left:4px solid #8F1734}
.small{font-size:12px;color:#64748b}
</style></head><body>
<h1>{{ title }}</h1>
<p class="small">Сформировано Banking Experiment Calculator. Отчёт не заменяет независимую валидацию и утверждённый статистический протокол.</p>
{% if passport %}<h2>Паспорт пилота</h2><table><tbody>{% for k,v in passport.items() %}<tr><th>{{ k }}</th><td>{{ v }}</td></tr>{% endfor %}</tbody></table>{% endif %}
{% for section in sections %}<h2>{{ section.title }}</h2>
{% if section.text %}<div class="note">{{ section.text }}</div>{% endif %}
{% if section.html %}{{ section.html | safe }}{% endif %}
{% endfor %}
{% if recommendations %}<h2>Выводы и ограничения</h2><ol>{% for x in recommendations %}<li>{{ x }}</li>{% endfor %}</ol>{% endif %}
</body></html>""")


def _safe_df(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, Mapping):
        return pd.DataFrame([value])
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        return pd.DataFrame(value)
    return pd.DataFrame()


def result_sections(results: Mapping[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    sections: list[dict[str, str]] = []
    recs: list[str] = []
    for analysis_name, payload in results.items():
        if not isinstance(payload, Mapping):
            continue
        added = False
        for key, value in payload.items():
            frame = _safe_df(value)
            if not frame.empty and len(frame.columns) <= 30:
                sections.append({
                    "title": f"{analysis_name}: {key}",
                    "text": "",
                    "html": frame.head(500).to_html(index=False, border=0, float_format=lambda x: f"{x:.6g}"),
                })
                added = True
        warnings = payload.get("warnings", [])
        if isinstance(warnings, list):
            recs.extend(str(x) for x in warnings)
        if not added:
            sections.append({"title": analysis_name, "text": "Результат сохранён в приложении, но не содержит табличной сводки.", "html": ""})
    return sections, recs


def build_html_protocol(
    *,
    title: str,
    passport: Mapping[str, Any] | None,
    results: Mapping[str, Any],
    recommendations: list[str] | None = None,
) -> bytes:
    sections, auto_recs = result_sections(results)
    payload = HTML_TEMPLATE.render(
        title=html.escape(title),
        passport={html.escape(str(k)): html.escape(str(v)) for k, v in (passport or {}).items()},
        sections=sections,
        recommendations=(recommendations or []) + auto_recs,
    )
    return payload.encode("utf-8")


def _register_cyrillic_font() -> str:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            if "DejaVuSans" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("DejaVuSans", path))
            return "DejaVuSans"
    return "Helvetica"


def build_pdf_protocol(
    *,
    title: str,
    passport: Mapping[str, Any] | None,
    results: Mapping[str, Any],
    recommendations: list[str] | None = None,
) -> bytes:
    """Формирует компактный PDF-протокол. Большие исходные таблицы в PDF не включаются."""
    font = _register_cyrillic_font()
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=14*mm, leftMargin=14*mm, topMargin=14*mm, bottomMargin=14*mm)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("BodyRU", parent=styles["BodyText"], fontName=font, fontSize=9, leading=12)
    h1 = ParagraphStyle("H1RU", parent=styles["Title"], fontName=font, fontSize=17, textColor=colors.HexColor("#8F1734"), alignment=TA_LEFT)
    h2 = ParagraphStyle("H2RU", parent=styles["Heading2"], fontName=font, fontSize=12, textColor=colors.HexColor("#243447"), spaceBefore=10)
    header = ParagraphStyle("HeaderRU", parent=body, fontName=font, fontSize=8, textColor=colors.white, leading=10)
    story = [Paragraph(title, h1), Spacer(1, 4*mm), Paragraph(
        "Автоматически сформированный протокол. Перед принятием решения проверьте предпосылки метода, качество данных и утверждённые правила пилота.", body
    )]

    if passport:
        story += [Paragraph("Паспорт пилота", h2)]
        data = [[Paragraph("Параметр", header), Paragraph("Значение", header)]] + [
            [Paragraph(str(k), body), Paragraph(str(v), body)] for k, v in passport.items()
        ]
        table = Table(data, colWidths=[48*mm, 120*mm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#243447")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("FONTNAME", (0,0), (-1,-1), font),
            ("FONTSIZE", (0,0), (-1,-1), 8),
        ]))
        story += [table]

    sections, auto_recs = result_sections(results)
    for section in sections:
        # В PDF используем исходный DataFrame из results, не HTML.
        analysis_name, _, key = section["title"].partition(": ")
        frame = _safe_df(results.get(analysis_name, {}).get(key))
        if frame.empty:
            continue
        frame = frame.head(30).iloc[:, :10]
        story += [Paragraph(section["title"], h2)]
        data = [[Paragraph(str(c), header) for c in frame.columns]]
        for _, row in frame.iterrows():
            data.append([Paragraph(str(v)[:250], body) for v in row])
        widths = [168*mm / max(len(frame.columns), 1)] * len(frame.columns)
        table = Table(data, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#243447")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("FONTNAME", (0,0), (-1,-1), font),
            ("FONTSIZE", (0,0), (-1,-1), 6.5),
        ]))
        story += [table]

    recs = (recommendations or []) + auto_recs
    if recs:
        story += [Paragraph("Выводы и ограничения", h2)]
        for i, x in enumerate(recs, 1):
            story.append(Paragraph(f"{i}. {x}", body))
    doc.build(story)
    return buffer.getvalue()


def build_advanced_excel_report(
    *,
    passport: Mapping[str, Any] | None,
    results: Mapping[str, Any],
    recommendations: list[str] | None = None,
) -> bytes:
    """Выгружает все табличные результаты advanced-модулей в единый XLSX."""
    from io import BytesIO
    buffer = BytesIO()
    used: set[str] = set()

    def sheet_name(raw: str) -> str:
        base = "".join(ch if ch not in "[]:*?/\\" else "_" for ch in raw)[:31] or "Sheet"
        name = base
        i = 2
        while name in used:
            suffix = f"_{i}"
            name = base[:31-len(suffix)] + suffix
            i += 1
        used.add(name)
        return name

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        book = writer.book
        title_fmt = book.add_format({"bold": True, "font_size": 15, "font_color": "#FFFFFF", "bg_color": "#8F1734"})
        header_fmt = book.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#243447", "border": 1})
        wrap = book.add_format({"text_wrap": True, "valign": "top"})

        if passport:
            frame = pd.DataFrame([{"Параметр": k, "Значение": v} for k, v in passport.items()])
            name = sheet_name("Паспорт")
            frame.to_excel(writer, sheet_name=name, index=False, startrow=2)
            ws = writer.sheets[name]
            ws.merge_range(0, 0, 0, 1, "Паспорт анализа", title_fmt)
            ws.set_row(2, 22, header_fmt)
            ws.set_column(0, 0, 30)
            ws.set_column(1, 1, 80, wrap)

        for analysis_name, payload in results.items():
            if not isinstance(payload, Mapping):
                continue
            for key, value in payload.items():
                frame = _safe_df(value)
                if frame.empty:
                    continue
                # Ограничиваем объём только для Excel-интерфейса: исходные assignments/curves всё равно можно выгрузить.
                name = sheet_name(f"{analysis_name}_{key}")
                frame.to_excel(writer, sheet_name=name, index=False, startrow=2)
                ws = writer.sheets[name]
                ws.merge_range(0, 0, 0, max(0, len(frame.columns)-1), f"{analysis_name}: {key}", title_fmt)
                ws.set_row(2, 22, header_fmt)
                ws.freeze_panes(3, 0)
                ws.autofilter(2, 0, 2 + len(frame), max(0, len(frame.columns)-1))
                for i, col in enumerate(frame.columns):
                    sample = frame[col].head(300).astype(str)
                    width = min(max([len(str(col)), *sample.map(len).tolist()] or [12]) + 2, 45)
                    ws.set_column(i, i, max(width, 12), wrap)

        recs = list(recommendations or [])
        for payload in results.values():
            if isinstance(payload, Mapping) and isinstance(payload.get("warnings"), list):
                recs.extend(str(x) for x in payload["warnings"])
        if recs:
            frame = pd.DataFrame({"№": range(1, len(recs)+1), "Вывод / ограничение": recs})
            name = sheet_name("Выводы")
            frame.to_excel(writer, sheet_name=name, index=False, startrow=2)
            ws = writer.sheets[name]
            ws.merge_range(0, 0, 0, 1, "Выводы и ограничения", title_fmt)
            ws.set_row(2, 22, header_fmt)
            ws.set_column(0, 0, 8)
            ws.set_column(1, 1, 100, wrap)
    return buffer.getvalue()
