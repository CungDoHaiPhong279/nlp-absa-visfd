"""Gradio demo (Task 12): aspect-based sentiment for Vietnamese phone reviews.

Usage: python demo/app.py --phobert-model models/final_phobert__seed42 \
    --tfidf-model models/final_tfidf__seed42 [--share]
"""

import argparse
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from absa.inference import (  # noqa: E402
    ASPECT_VI,
    COLORS,
    POLARITY_VI,
    Predictor,
    render_html,
    summarize_predictions,
)

# From results/error_analysis/selected_examples.md: Dev comments the model gets (partly) wrong
EXAMPLES = [
    "Sp dùng rất tốt trong tầm giá tầm 2tr. "
    "Các tác vụ rất tốt không thua gì các máy tầm trung khác",
    "Máy ok chỉ mỗi tội màn ám vàng quá, màu đen bóng đẹp nhưng dể xước lông mèo. "
    "Màn hình ám vàng nên ko đẹp bằng dòng 6s và 6s plus.",
    "Nói chung thì ổn nhưng pin tụt hơi nhanh chơi k nóng máy . "
    "Camera trước thì k tự xoá phông phải xoá thủ công",
    "cho hỏi 9S còn bị lỗi camera trước không ạ. "
    "đang tính mua 1 con nhưng thấy bị lỗi camera lỗi quá",
    "Cấu hình quá thấp. Không có gì đặc biệt. Chỉ là giá tiền của thương hiệu, Mà thương hiệu này "
    "bây giờ cũng không còn nóng như trước. "
    "Với giá tiền đấy mua được rất nhiều loại điện thoại khác "
    "có cấu hình mạnh gấp đôi gấp ba.",
]

CSS = """
.gradio-container{max-width:1220px !important;width:100% !important;margin:auto;
  background:#eef3ef;padding:0 30px 28px !important}
body{background:#eef3ef !important}
footer{display:none !important}
.hero{display:flex;align-items:stretch;justify-content:space-between;gap:28px;padding:26px 30px;
  margin:18px 0 16px;background:#fff;border:1px solid #d7e2da;
  border-radius:10px;box-shadow:0 4px 12px rgba(28,55,38,.06)}
.hero-brand{display:block}
.brand-mark{display:none}
.eyebrow{font-size:11px;letter-spacing:1.8px;text-transform:uppercase;font-weight:800;color:#16803f;margin-bottom:8px}
.hero h1{font-size:30px;font-weight:760;margin:0;letter-spacing:-.5px;color:#173b28}
.hero p{max-width:700px;margin:11px 0 0;color:#64756a;font-size:15px;line-height:1.5}
.hero-stats{display:grid;grid-template-columns:repeat(3,auto);align-content:center;gap:0;
  min-width:330px;border-left:1px solid #dfe8e1}
.hero-stat{padding:0 20px;border-right:1px solid #dfe8e1;text-align:center}
.hero-stat:last-child{border-right:0}
.hero-stat .num{font-size:24px;font-weight:780;color:#16803f;line-height:1.1}
.hero-stat .label{font-size:11px;color:#728277;margin-top:5px;white-space:nowrap}
.model-tabs,.feature-tabs{background:#fff;border-radius:10px;padding:0 12px;
  border:1px solid #d7e2da;box-shadow:0 3px 10px rgba(28,55,38,.045)}
.model-tabs{padding-bottom:20px !important}
.feature-tabs{margin-top:14px !important;margin-bottom:4px !important;padding-bottom:16px !important}
.model-tabs > .tab-nav,.feature-tabs > .tab-nav{gap:8px !important;padding:7px 6px 0 !important}
.model-tabs > .tab-nav button,.feature-tabs > .tab-nav button{
  border:1px solid transparent !important;border-radius:8px 8px 0 0 !important;
  margin:0 2px 4px !important;transition:background .15s ease,border-color .15s ease}
.model-tabs > .tab-nav button{font-size:16px;font-weight:750;padding:12px 20px}
.feature-tabs > .tab-nav button{font-weight:650;padding:11px 18px}
.model-tabs .tab-nav button.selected,.model-tabs .tab-nav button[aria-selected="true"],
.feature-tabs .tab-nav button.selected,.feature-tabs .tab-nav button[aria-selected="true"]{
  background:#edf7ef !important;color:#176b38 !important;border-color:#cce4d2 !important;
  border-bottom:3px solid #16803f !important}
.panel-card{background:#fbfefb !important;border:1px solid #d7e2da !important;
  border-radius:10px !important;box-shadow:0 3px 10px rgba(28,55,38,.045) !important;padding:18px !important}
.result-card{background:#f5faf6 !important;border:1px solid #d7e2da !important;
  border-radius:10px !important;padding:18px !important;margin-top:18px !important}
.panel-card{height:max-content !important;align-self:flex-start !important}
.comment-input-card,.comment-result-card{background:transparent !important;border:0 !important;
  box-shadow:none !important;padding:10px 8px !important}
.comment-result-card{margin-top:0 !important}
.gradio-container textarea{border-radius:13px !important;border-color:#d8e3f0 !important;
  box-shadow:inset 0 1px 3px rgba(15,23,42,.03) !important;font-size:15px !important}
.gradio-container button.primary{border-radius:8px !important;font-weight:750 !important;
  background:#16803f !important;box-shadow:0 4px 9px rgba(22,128,63,.20) !important}
.gradio-container button.primary:hover{background:#116b35 !important;transform:translateY(-1px)}
.gradio-container .upload-container{border-radius:14px !important;border-color:#cbd9eb !important}
.res-empty{padding:44px 16px;text-align:center;color:var(--body-text-color-subdued);
  border:1px dashed var(--border-color-primary);border-radius:14px;font-size:14px}
.res-empty .big{font-size:17px;font-weight:600;color:var(--body-text-color);margin-bottom:4px}
.res-empty.slim{padding:18px}
.quote{font-size:16px;line-height:1.65;padding:14px 18px;border-left:3px solid var(--color-accent);
  background:var(--background-fill-secondary);border-radius:0 12px 12px 0;margin-bottom:18px;
  white-space:pre-wrap}
.verdict{margin-bottom:14px}
.vt{font-size:14px;margin-bottom:8px;color:var(--body-text-color-subdued)}
.strip{display:flex;height:8px;border-radius:99px;overflow:hidden;gap:3px}
.strip span{display:block;height:100%;border-radius:99px}
.tags{display:flex;flex-wrap:wrap;gap:10px}
.tag{font-size:14px;box-shadow:0 1px 2px rgba(0,0,0,.08)}
.rest{margin-top:14px;font-size:13px;color:var(--body-text-color-subdued)}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}
.kpi{padding:15px 16px;border-radius:10px;background:#f8fcf9;
  border:1px solid #dbe9de;border-top:3px solid #b8d9c1;box-shadow:0 2px 8px rgba(28,55,38,.04)}
.kpi .l{font-size:13px;color:var(--body-text-color-subdued)}
.kpi .x{font-size:24px;font-weight:700;margin-top:4px}
.section-title{font-size:16px;font-weight:700;margin:18px 0 8px}
.table-note{font-size:13px;color:var(--body-text-color-subdued);margin:8px 0 12px}
.alert{padding:10px 14px;border-radius:10px;background:rgba(207,34,46,.10);color:#cf222e;
  font-size:14px;font-weight:600;margin-bottom:10px}
.note{font-size:13px;color:var(--body-text-color-subdued);margin-bottom:10px}
.foot{font-size:12px;color:#8293aa;text-align:center;padding:24px 0 8px}
@media (max-width:850px){.kpis{grid-template-columns:repeat(2,1fr)}
  .hero{display:block}.hero-stats{min-width:0;margin-top:20px;padding-top:16px;
    border-left:0;border-top:1px solid #dfe8e1}.hero-stat{padding:0 12px}}
@media (max-width:640px){.kpis{grid-template-columns:1fr}.hero h1{font-size:24px}
  .tab-container button{white-space:normal !important;flex:1 1 0;min-width:0}}
"""

HERO = (
    '<div class="hero"><div class="hero-brand"><div class="eyebrow">ABSA · UIT-ViSFD</div>'
    '<h1>Đọc vị bình luận điện thoại</h1>'
    "<p>Dán một bình luận, mô hình sẽ nhận diện khách đang khen hay chê "
    "<b>từng khía cạnh</b> của sản phẩm.</p></div>"
    '<div class="hero-stats"><div class="hero-stat"><div class="num">10</div>'
    '<div class="label">khía cạnh</div></div><div class="hero-stat"><div class="num">3</div>'
    '<div class="label">mức cảm xúc</div></div><div class="hero-stat"><div class="num">2</div>'
    '<div class="label">mô hình</div></div></div></div>'
)
PLACEHOLDER = (
    '<div class="res-empty"><div class="big">Kết quả sẽ hiện ở đây</div>'
    "Dán một bình luận hoặc chọn một ví dụ có sẵn.</div>"
)
CSV_PLACEHOLDER = '<div class="res-empty">Tải lên file CSV có cột <code>comment</code>.</div>'


def make_theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue="green",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Be Vietnam Pro"), "system-ui", "sans-serif"],
    )


def plot_summary(table: pd.DataFrame, figsize=None):
    """Horizontal stacked bars: Positive/Neutral/Negative count per mentioned aspect."""
    t = table[table["Total"] > 0].sort_values("Total")
    fig, ax = plt.subplots(figsize=figsize or (7.5, 0.45 * max(len(t), 1) + 1.2))
    left = pd.Series(0, index=t.index)
    for p in ("Positive", "Neutral", "Negative"):
        ax.barh(
            [ASPECT_VI[a] for a in t.index],
            t[p],
            left=left,
            color=COLORS[p],
            label=POLARITY_VI[p],
            height=0.62,
        )
        left = left + t[p]
    # transparent background + mid-gray text: readable in both light and dark Gradio themes
    gray = "#8b949e"
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(gray)
    ax.tick_params(axis="y", length=0, colors=gray)
    ax.tick_params(axis="x", colors=gray)
    ax.set_xlabel("Số lần được nhắc", color=gray)
    # counts: integer ticks on round steps (no 0.25, no 150)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    # top-right: the top-left corner is covered by the gr.Plot label
    legend = ax.legend(ncol=3, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    for text in legend.get_texts():
        text.set_color(gray)
    fig.tight_layout()
    return fig


def read_comments(file, max_rows: int) -> pd.DataFrame:
    """Read at most max_rows + 1 rows (to detect truncation); ',' or ';' separated, UTF-8."""
    try:
        with open(file, encoding="utf-8-sig") as fh:
            header = fh.readline()
        sep = ";" if header.count(";") > header.count(",") else ","  # Excel VN saves with ';'
        # keep_default_na=False: a comment that is literally "NA" or "None" stays text
        df = pd.read_csv(
            file, sep=sep, nrows=max_rows + 1, encoding="utf-8-sig", keep_default_na=False
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as err:
        raise gr.Error("Không đọc được file: cần file CSV mã hóa UTF-8, có dòng tiêu đề.") from err
    if "comment" not in df.columns:
        raise gr.Error("File CSV cần có cột comment.")
    return df


def detailed_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return an explicit count/rate table for the dashboard and report."""
    rows = []
    for aspect in table.index:
        total = int(table.loc[aspect, "Total"])
        if total == 0:
            continue
        row = {
            "Khía cạnh": ASPECT_VI[aspect],
            "Tổng lượt nhắc": total,
        }
        for polarity, label in POLARITY_VI.items():
            count = int(table.loc[aspect, polarity])
            row[f"{label} (lượt)"] = count
            row[f"{label} (%)"] = round(100 * count / total, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def export_report_files(
    table: pd.DataFrame,
    shown: pd.DataFrame,
    chart,
    model_label: str,
    source_name: str,
) -> tuple[str, str]:
    """Create downloadable CSV and PDF artifacts in the system temp directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_label.lower().replace("-", "").replace(" ", "_")
    safe_source = Path(source_name).stem.replace(" ", "_")
    csv_path = Path(tempfile.gettempdir()) / f"absa_{safe_model}_{safe_source}_{stamp}.csv"
    pdf_path = Path(tempfile.gettempdir()) / f"absa_{safe_model}_{safe_source}_{stamp}.pdf"
    shown.to_csv(csv_path, index=False, encoding="utf-8-sig")

    total_comments = int(table.attrs.get("n_comments", 0))
    total_mentions = int(table["Total"].sum())
    with PdfPages(pdf_path) as pdf:
        page, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.text(0.03, 0.91, "Báo cáo phân tích bình luận", fontsize=24, weight="bold", color="#173b28")
        ax.text(0.03, 0.855, "UIT-ViSFD · Phân tích cảm xúc theo khía cạnh", fontsize=12, color="#64756a")
        ax.text(0.03, 0.78, f"Mô hình: {model_label}", fontsize=13, weight="bold")
        ax.text(0.03, 0.735, f"Tệp dữ liệu: {source_name}", fontsize=11)
        ax.text(0.03, 0.695, f"Số bình luận: {total_comments:,} · Tổng lượt khía cạnh: {total_mentions:,}".replace(",", "."), fontsize=11)
        ax.text(0.03, 0.61, "Bảng dưới đây thể hiện số lượt và tỷ lệ cảm xúc trên từng khía cạnh.", fontsize=11, color="#64756a")
        pdf.savefig(page, bbox_inches="tight")
        plt.close(page)

        chart_pdf = plot_summary(table, figsize=(11.69, 8.27))
        pdf.savefig(chart_pdf)
        plt.close(chart_pdf)

        table_page, table_ax = plt.subplots(figsize=(14, max(5, 0.45 * len(shown) + 2)))
        table_ax.axis("off")
        table_ax.set_title("Chi tiết theo khía cạnh", loc="left", fontsize=16, weight="bold", pad=16)
        values = shown.astype(str).values.tolist()
        table_artist = table_ax.table(
            cellText=values,
            colLabels=shown.columns.tolist(),
            loc="upper center",
            cellLoc="center",
            colLoc="center",
        )
        table_artist.auto_set_font_size(False)
        table_artist.set_fontsize(9)
        table_artist.scale(1, 1.65)
        pdf.savefig(table_page, bbox_inches="tight")
        plt.close(table_page)
    return str(csv_path), str(pdf_path)


def build_app(
    predictor: Predictor,
    max_rows: int = 2000,
    tfidf_predictor=None,
) -> gr.Blocks:
    """Build the demo; when supplied, show TF-IDF and PhoBERT side by side."""

    def add_model_tab(label: str, model) -> None:
        def analyze(text: str) -> str:
            if not text or not text.strip():
                return PLACEHOLDER
            return render_html(text, model.predict(text))

        def summarize(file):
            if file is None:
                return CSV_PLACEHOLDER, None, None, None, None
            df = read_comments(file, max_rows)
            truncated = len(df) > max_rows
            comments = df["comment"].fillna("").astype(str).head(max_rows).tolist()
            preds = model.predict_batch(comments)
            table = summarize_predictions(preds)
            table.attrs["n_comments"] = len(preds)
            note = (
                f'<div class="note">File có hơn {max_rows:,} dòng; chỉ phân tích {max_rows:,} dòng đầu.'
                "</div>"
                if truncated
                else ""
            ).replace(",", ".")
            mentioned = 100 * sum(bool(d) for d in preds) / max(len(preds), 1)
            total_mentions = int(table["Total"].sum())
            total_positive = int(table["Positive"].sum())
            total_neutral = int(table["Neutral"].sum())
            total_negative = int(table["Negative"].sum())
            worst = table["Negative"].idxmax() if table["Negative"].max() > 0 else None
            kpi = (
                f'<div class="kpis"><div class="kpi"><div class="l">Bình luận đã phân tích</div>'
                + f'<div class="x">{len(preds):,}</div></div>'.replace(",", ".")
                + f'<div class="kpi"><div class="l">Có nhắc ít nhất 1 khía cạnh</div>'
                + f'<div class="x">{mentioned:.0f}%</div></div>'
                + f'<div class="kpi"><div class="l">Tổng lượt khía cạnh</div>'
                + f'<div class="x">{total_mentions:,}</div></div>'.replace(",", ".")
                + f'<div class="kpi"><div class="l">Bị chê nhiều nhất</div>'
                + f'<div class="x" style="color:#cf222e">{ASPECT_VI[worst] if worst else "—"}</div>'
                "</div></div>"
            )
            alert = ""
            if worst:
                neg, tot = int(table.loc[worst, "Negative"]), int(table.loc[worst, "Total"])
                alert = (
                    f'<div class="alert">Khách chê {ASPECT_VI[worst].lower()} nhiều nhất: '
                    f"{neg} lần, chiếm {100 * neg / tot:.0f}% số lần được nhắc.</div>"
                )
            shown = detailed_table(table)
            sentiment_note = (
                f'<div class="table-note"><b>Phân bố toàn bộ lượt nhắc:</b> '
                + f'Tích cực {total_positive:,} · Trung tính {total_neutral:,} · '
                + f'Tiêu cực {total_negative:,}.'.replace(",", ".")
                + " Tỷ lệ trong bảng được tính riêng trên từng khía cạnh.</div>"
            )
            chart = plot_summary(table)
            csv_path, pdf_path = export_report_files(
                table, shown, chart, label, Path(file).name
            )
            return note + kpi + alert + sentiment_note, chart, shown, csv_path, pdf_path

        with gr.Tab(label):
            with gr.Tabs(elem_classes=["feature-tabs"]):
              with gr.Tab("Phân tích một bình luận"):
                out = gr.HTML(PLACEHOLDER, render=False)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes=["panel-card", "comment-input-card"]):
                        inp = gr.Textbox(
                            label="Bình luận",
                            lines=5,
                            placeholder="Ví dụ: Máy đẹp nhưng pin yếu, giá hơi chát",
                        )
                        btn = gr.Button("Phân tích", variant="primary", size="lg")
                        gr.Examples(
                            EXAMPLES,
                            inputs=inp,
                            outputs=out,
                            fn=analyze,
                            run_on_click=True,
                            label="Thử nhanh (bình luận thật trong tập Dev)",
                        )
                    with gr.Column(scale=7, elem_classes=["result-card", "comment-result-card"]):
                        out.render()
                btn.click(analyze, inp, out)
                inp.submit(analyze, inp, out)
              with gr.Tab("Tổng hợp từ file CSV"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, elem_classes=["panel-card"]):
                        f = gr.File(label="File CSV (cần cột comment)", file_types=[".csv"])
                        gr.Markdown(f"Tối đa {max_rows:,} dòng đầu được phân tích.".replace(",", "."))
                    with gr.Column(scale=8, elem_classes=["result-card"]):
                        summary = gr.HTML(CSV_PLACEHOLDER)
                        chart = gr.Plot(label="Cảm xúc theo khía cạnh")
                        table = gr.Dataframe(
                            label="Chi tiết theo khía cạnh",
                            interactive=False,
                            wrap=True,
                        )
                        with gr.Row():
                            csv_download = gr.File(label="Tải bảng CSV", interactive=False)
                            pdf_download = gr.File(label="Tải báo cáo PDF", interactive=False)
                f.change(summarize, f, [summary, chart, table, csv_download, pdf_download])

    with gr.Blocks(title="Đọc vị bình luận điện thoại") as demo:
        gr.HTML(HERO)
        with gr.Tabs(elem_classes=["model-tabs"]):
            if tfidf_predictor is not None:
                add_model_tab("TF-IDF", tfidf_predictor)
            add_model_tab("PhoBERT", predictor)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="legacy alias for the PhoBERT checkpoint")
    parser.add_argument("--phobert-model", type=Path, help="PhoBERT checkpoint directory")
    parser.add_argument("--tfidf-model", type=Path, help="TF-IDF checkpoint directory")
    parser.add_argument("--share", action="store_true", help="public gradio.live link")
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    phobert_path = args.phobert_model or args.model
    if phobert_path is None:
        parser.error("cần --phobert-model (hoặc --model)")
    phobert = Predictor.from_dir(phobert_path)
    tfidf = Predictor.from_dir(args.tfidf_model) if args.tfidf_model else None
    demo = build_app(phobert, args.max_rows, tfidf_predictor=tfidf)
    demo.launch(share=args.share, server_port=args.port, theme=make_theme(), css=CSS)


if __name__ == "__main__":
    main()
