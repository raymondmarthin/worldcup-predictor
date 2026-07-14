"""
make_progress1_simple.py
Regenerates progress-1.pdf with a plain, formal style:
Times New Roman, black text only, no colors, simple tables.
"""
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBW", fontName="Times-Bold", fontSize=16,
                           textColor=colors.black, spaceAfter=6))
styles.add(ParagraphStyle(name="SubBW", fontName="Times-Roman", fontSize=10,
                           textColor=colors.black, spaceAfter=12))
styles.add(ParagraphStyle(name="H2BW", fontName="Times-Bold", fontSize=12,
                           textColor=colors.black, spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle(name="BodyBW", fontName="Times-Roman", fontSize=11,
                           textColor=colors.black, leading=15, spaceAfter=8))


def simple_table_style():
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, styles["BodyBW"]), bulletColor=colors.black) for t in items],
        bulletType="bullet",
    )


def build():
    results = pd.read_csv(ROOT / "data" / "processed" / "processed_matches.csv", parse_dates=["date"])
    rankings = pd.read_csv(ROOT / "data" / "processed" / "processed_rankings.csv", parse_dates=["date"])
    wc2026 = pd.read_csv(ROOT / "data" / "processed" / "processed_wc2026.csv", parse_dates=["date"])
    feat = pd.read_csv(ROOT / "data" / "processed" / "training_features.csv", parse_dates=["date"])

    doc = SimpleDocTemplate(str(ROOT / "progress-1.pdf"), pagesize=letter,
                             topMargin=2.2*cm, bottomMargin=2.2*cm, leftMargin=2.2*cm, rightMargin=2.2*cm)
    story = []

    story.append(Paragraph("World Cup 2026 Live Knockout Predictor API", styles["SubBW"]))
    story.append(Paragraph("Laporan Progress 1: Data &amp; Feature Engineering", styles["TitleBW"]))
    story.append(Paragraph("Milestone: Hari 1-3 &mdash; pengumpulan data, cleaning, canonical team mapping, "
                            "feature engineering", styles["SubBW"]))

    story.append(Paragraph("1. Apa yang dikerjakan", styles["H2BW"]))
    story.append(bullets([
        "Mengunduh 3 dataset mentah: hasil pertandingan internasional (1872-sekarang), riwayat "
        "FIFA ranking (Des 1992-Sep 2024), dan hasil pertandingan World Cup 2026 terkini.",
        "Membuat canonical team-name mapping (src/clean.py) untuk menyelaraskan ketidakkonsistenan "
        "nama tim antar sumber, contoh: \"Cabo Verde\" menjadi \"Cape Verde\", \"T&uuml;rkiye\" "
        "menjadi \"Turkey\", \"USA\" menjadi \"United States\".",
        "Membersihkan dan mengetipkan seluruh raw CSV ke data/processed/ murni lewat script (raw "
        "data tidak pernah diedit manual, sesuai ketentuan tugas).",
        "Mengimplementasikan feature engineering (src/features.py) dengan aturan ketat anti-data-"
        "leakage: setiap fitur hanya memakai informasi sebelum tanggal pertandingan, dibangun "
        "lewat satu pass kronologis per tim.",
        "Mengimplementasikan resolusi hasil draw pada babak knockout memakai data adu penalti, "
        "sehingga pertandingan knockout historis yang berakhir imbang di 90/120 menit tetap "
        "mendapat label pemenang yang jelas.",
    ]))

    story.append(Paragraph("2. Ringkasan dataset", styles["H2BW"]))
    data = [
        ["Dataset", "Baris", "Rentang tanggal", "Peran"],
        ["international_results.csv", f"{len(results):,}", "1872 - 2026", "Training / backtest"],
        ["fifa_ranking_historical.csv", f"{len(rankings):,}", "1992-12 - 2024-09", "Proxy elo/ranking"],
        ["wc2026_matches_detailed.csv", f"{len(wc2026):,}", "2026-06 - 2026-07", "Konteks live inference saja"],
        ["training_features.csv (hasil olahan)", f"{len(feat):,}", "1993 - 2026", "Input training model"],
    ]
    t = Table(data, colWidths=[6.5*cm, 2.2*cm, 3.3*cm, 4.3*cm])
    t.setStyle(simple_table_style())
    story.append(t)

    story.append(Paragraph("3. Feature set yang diimplementasikan", styles["H2BW"]))
    feat_data = [
        ["Fitur", "Prioritas", "Status"],
        ["elo_delta", "Wajib", "Selesai"],
        ["rank_delta", "Wajib", "Selesai"],
        ["recent_form_delta", "Wajib", "Selesai (10 pertandingan terakhir)"],
        ["goal_diff_recent_delta", "Wajib", "Selesai (10 pertandingan terakhir)"],
        ["world_cup_2026_goal_diff_delta", "Wajib (live)", "Selesai, khusus inference"],
        ["rest_days_delta", "Opsional", "Selesai"],
        ["h2h_delta", "Opsional", "Belum diimplementasikan (didokumentasikan di REPORT.md)"],
        ["penalty_history_delta", "Bonus", "Belum diimplementasikan"],
    ]
    t2 = Table(feat_data, colWidths=[6.3*cm, 3*cm, 6.9*cm])
    t2.setStyle(simple_table_style())
    story.append(t2)

    story.append(Paragraph("4. Temuan", styles["H2BW"]))
    story.append(bullets([
        f"Setelah difilter ke pertandingan dengan cakupan FIFA ranking (mulai 1993) dan pemenang "
        f"yang jelas, feature set training memiliki {len(feat):,} baris dengan distribusi target "
        f"{int(feat['target'].mean()*100)}% kemenangan team_a (wajar karena keunggulan tim tuan "
        f"rumah pada dataset ini).",
        "138 nama tim pada data historis (sebagian besar mikronegara/wilayah non-FIFA) tidak "
        "memiliki padanan di data FIFA ranking. Ini wajar dan tidak memengaruhi 48 tim resmi "
        "WC2026.",
    ]))

    story.append(Paragraph("5. Kendala", styles["H2BW"]))
    story.append(bullets([
        "Dataset FIFA ranking berhenti di Sep 2024; tidak ada sumber data ranking terstruktur "
        "gratis untuk 2025-2026 yang bisa diakses di environment ini. Diatasi dengan memakai "
        "snapshot ranking terakhir yang tersedia (forward-fill) dan menambahkan fitur live "
        "goal-difference WC2026.",
        "Tidak ada label \"babak knockout\" eksplisit pada data pertandingan historis, sehingga "
        "model belajar dari seluruh pertandingan internasional yang memiliki pemenang jelas, "
        "bukan khusus pertandingan knockout. Hal ini didokumentasikan sebagai limitasi di "
        "REPORT.md.",
    ]))

    doc.build(story)
    print("progress-1.pdf berhasil dibuat ulang (Bahasa Indonesia, Times New Roman, hitam, sederhana)")


if __name__ == "__main__":
    build()
