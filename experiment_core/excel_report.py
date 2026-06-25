from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


def _safe_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        return pd.DataFrame([value])
    if isinstance(value, list):
        if not value:
            return pd.DataFrame()
        if isinstance(value[0], dict):
            return pd.DataFrame(value)
        return pd.DataFrame({"value": value})
    return pd.DataFrame({"value": [value]})


def build_excel_report(
    *,
    passport: dict[str, Any],
    design: pd.DataFrame | dict[str, Any] | None = None,
    scenarios: pd.DataFrame | None = None,
    analysis: dict[str, Any] | None = None,
    recommendations: list[str] | None = None,
    uplift: dict[str, Any] | None = None,
) -> bytes:
    """Формирует переносимый Excel-файл без сохранения данных в БД."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        workbook = writer.book
        title_fmt = workbook.add_format({
            "bold": True, "font_size": 16, "font_color": "#FFFFFF",
            "bg_color": "#8F1734", "align": "left", "valign": "vcenter",
        })
        header_fmt = workbook.add_format({
            "bold": True, "font_color": "#FFFFFF", "bg_color": "#243447",
            "border": 1, "align": "center", "valign": "vcenter",
        })
        percent_fmt = workbook.add_format({"num_format": "0.00%"})
        number_fmt = workbook.add_format({"num_format": "0.0000"})
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

        passport_df = pd.DataFrame(
            [{"Параметр": key, "Значение": value} for key, value in passport.items()]
        )
        passport_df.to_excel(writer, sheet_name="Паспорт", index=False, startrow=2)
        ws = writer.sheets["Паспорт"]
        ws.merge_range("A1:B1", "Паспорт пилота", title_fmt)
        ws.set_row(0, 26)
        ws.set_column("A:A", 30)
        ws.set_column("B:B", 70, wrap_fmt)
        ws.set_row(2, 22, header_fmt)

        def write_df(sheet_name: str, title: str, frame: pd.DataFrame | None) -> None:
            frame = _safe_frame(frame)
            if frame.empty:
                return
            frame.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
            sheet = writer.sheets[sheet_name]
            last_col = max(0, len(frame.columns) - 1)
            sheet.merge_range(0, 0, 0, last_col, title, title_fmt)
            sheet.set_row(2, 22, header_fmt)
            sheet.freeze_panes(3, 0)
            sheet.autofilter(2, 0, 2 + len(frame), last_col)
            for i, col in enumerate(frame.columns):
                max_len = max(len(str(col)), *(len(str(v)) for v in frame[col].head(200)))
                sheet.set_column(i, i, min(max(max_len + 2, 12), 40), wrap_fmt)

        write_df("Дизайн", "Расчёт дизайна", _safe_frame(design))
        write_df("Сценарии", "Сценарии сокращения срока", scenarios)

        if analysis:
            write_df("Группы", "Сводка по экспериментальным группам", analysis.get("group_summary"))
            write_df("Результаты", "Статистические результаты", analysis.get("results"))
            write_df("Качество", "Проверки качества данных", analysis.get("quality"))

        if uplift:
            write_df("Uplift", "Калибровка uplift-модели", uplift.get("calibration"))

        recs = recommendations or []
        if analysis:
            recs = recs + list(analysis.get("warnings", []))
        if uplift:
            recs = recs + list(uplift.get("warnings", []))
        if recs:
            rec_df = pd.DataFrame({"№": range(1, len(recs) + 1), "Рекомендация": recs})
            write_df("Рекомендации", "Выводы и ограничения", rec_df)

    return buffer.getvalue()
