"""Gradio demo (Task 12): aspect-based sentiment for Vietnamese phone reviews.

Usage: python demo/app.py --model models/final_phobert__seed42 [--share] [--max-rows 2000]
"""

import argparse
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from absa.inference import (  # noqa: E402
    ASPECT_VI,
    COLORS,
    POLARITY_VI,
    Predictor,
    render_html,
    summarize_predictions,
)

# From results/error_analysis/selected_examples.md (Dev comments the model gets wrong or right)
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
.gradio-container{max-width:1120px !important;width:100% !important;margin:auto}
footer{display:none !important}
.hero{padding:28px 4px 8px}
.hero h1{font-size:30px;font-weight:700;margin:0 0 6px;letter-spacing:-.3px}
.hero p{margin:0;color:var(--body-text-color-subdued);font-size:15px}
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
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px}
.kpi{padding:14px 16px;border-radius:12px;background:var(--background-fill-secondary)}
.kpi .l{font-size:13px;color:var(--body-text-color-subdued)}
.kpi .x{font-size:24px;font-weight:700;margin-top:4px}
.alert{padding:10px 14px;border-radius:10px;background:rgba(207,34,46,.10);color:#cf222e;
  font-size:14px;font-weight:600;margin-bottom:10px}
.note{font-size:13px;color:var(--body-text-color-subdued);margin-bottom:10px}
.foot{font-size:12px;color:var(--body-text-color-subdued);text-align:center;padding:18px 0 6px}
@media (max-width:640px){.kpis{grid-template-columns:1fr}.hero h1{font-size:24px}
  .tab-container button{white-space:normal !important;flex:1 1 0;min-width:0}}
"""

HERO = (
    '<div class="hero"><h1>Đọc vị bình luận điện thoại</h1>'
    "<p>Dán một bình luận, mô hình PhoBERT cho biết khách đang khen hay chê "
    "<b>từng khía cạnh</b>: pin, camera, giá, dịch vụ…</p></div>"
)
PLACEHOLDER = (
    '<div class="res-empty"><div class="big">Kết quả sẽ hiện ở đây</div>'
    "Dán một bình luận hoặc chọn một ví dụ bên trái.</div>"
)
CSV_PLACEHOLDER = '<div class="res-empty">Tải lên file CSV có cột <code>comment</code>.</div>'


def make_theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Be Vietnam Pro"), "system-ui", "sans-serif"],
    )


def plot_summary(table: pd.DataFrame):
    """Horizontal stacked bars: Positive/Neutral/Negative count per mentioned aspect."""
    t = table[table["Total"] > 0].sort_values("Total")
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * max(len(t), 1) + 1.2))
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
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # counts, no 0.25 ticks
    # top-right: the top-left corner is covered by the gr.Plot label
    legend = ax.legend(ncol=3, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    for text in legend.get_texts():
        text.set_color(gray)
    fig.tight_layout()
    plt.close(fig)
    return fig


def read_comments(file, max_rows: int) -> pd.DataFrame:
    """Read at most max_rows + 1 rows (to detect truncation); ',' or ';' separated, UTF-8."""
    try:
        with open(file, encoding="utf-8-sig") as fh:
            header = fh.readline()
        sep = ";" if header.count(";") > header.count(",") else ","  # Excel VN saves with ';'
        df = pd.read_csv(file, sep=sep, nrows=max_rows + 1, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as err:
        raise gr.Error("Không đọc được file: cần file CSV mã hóa UTF-8, có dòng tiêu đề.") from err
    if "comment" not in df.columns:
        raise gr.Error("File CSV cần có cột 'comment'")
    return df


def build_app(predictor: Predictor, max_rows: int = 2000) -> gr.Blocks:
    def analyze(text: str) -> str:
        if not text or not text.strip():
            return PLACEHOLDER
        return render_html(text, predictor.predict(text))

    def summarize(file):
        if file is None:
            return CSV_PLACEHOLDER, None, None
        df = read_comments(file, max_rows)
        truncated = len(df) > max_rows
        comments = df["comment"].fillna("").astype(str).head(max_rows).tolist()
        preds = predictor.predict_batch(comments)
        table = summarize_predictions(preds)
        note = (
            f'<div class="note">File có hơn {max_rows:,} dòng; chỉ phân tích {max_rows:,} dòng đầu.'
            "</div>"
            if truncated
            else ""
        ).replace(",", ".")
        mentioned = 100 * sum(bool(d) for d in preds) / max(len(preds), 1)
        worst = table["Negative"].idxmax() if table["Negative"].max() > 0 else None
        kpi = (
            f'<div class="kpis"><div class="kpi"><div class="l">Bình luận đã phân tích</div>'
            f'<div class="x">{len(preds):,}</div></div>'.replace(",", ".")
            + f'<div class="kpi"><div class="l">Có nhắc ít nhất 1 khía cạnh</div>'
            f'<div class="x">{mentioned:.0f}%</div></div>'
            f'<div class="kpi"><div class="l">Bị chê nhiều nhất</div>'
            f'<div class="x" style="color:#cf222e">{ASPECT_VI[worst] if worst else "—"}</div>'
            "</div></div>"
        )
        alert = ""
        if worst:
            neg, tot = int(table.loc[worst, "Negative"]), int(table.loc[worst, "Total"])
            alert = (
                f'<div class="alert">Khách chê {ASPECT_VI[worst].lower()} nhiều nhất: '
                f"{neg} lần, chiếm {100 * neg / tot:.0f}% số lần được nhắc.</div>"
            )
        shown = table[table["Total"] > 0].rename(index=ASPECT_VI, columns=POLARITY_VI)
        shown = shown.rename(columns={"Total": "Tổng"}).reset_index(names="Khía cạnh")
        return note + kpi + alert, plot_summary(table), shown

    with gr.Blocks(title="Đọc vị bình luận điện thoại") as demo:
        gr.HTML(HERO)
        with gr.Tabs():
            with gr.Tab("Phân tích một bình luận"):
                out = gr.HTML(PLACEHOLDER, render=False)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
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
                    with gr.Column(scale=7):
                        out.render()
                btn.click(analyze, inp, out)
                inp.submit(analyze, inp, out)
            with gr.Tab("Tổng hợp từ file CSV"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4):
                        f = gr.File(label="File CSV (cần cột comment)", file_types=[".csv"])
                        gr.Markdown(
                            f"Tối đa {max_rows:,} dòng đầu được phân tích.".replace(",", ".")
                        )
                    with gr.Column(scale=8):
                        summary = gr.HTML(CSV_PLACEHOLDER)
                        chart = gr.Plot(label="Khen / chê theo khía cạnh")
                        table = gr.Dataframe(label="Bảng tổng hợp", interactive=False)
                f.change(summarize, f, [summary, chart, table])
        gr.HTML(
            '<div class="foot">Bài tập lớn NLP · Dữ liệu UIT-ViSFD (Phan et al., 2021) · '
            "Chỉ dùng cho nghiên cứu</div>"
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="checkpoint directory")
    parser.add_argument("--share", action="store_true", help="public gradio.live link")
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    demo = build_app(Predictor.from_dir(args.model), args.max_rows)
    demo.launch(share=args.share, server_port=args.port, theme=make_theme(), css=CSS)


if __name__ == "__main__":
    main()
