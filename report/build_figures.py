#!/usr/bin/env python3
"""Build figures used by report/report.tex."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Times.ttc",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    ):
        p = Path(candidate)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def draw_bar_panel(
    draw: ImageDraw.ImageDraw,
    title: str,
    labels: list[str],
    values: list[float],
    colors: list[tuple[int, int, int]],
    panel: tuple[int, int, int, int],
    value_fmt: str,
) -> None:
    x0, y0, w, h = panel
    margin_left = 72
    margin_bottom = 56
    margin_top = 48
    margin_right = 26

    font_title = load_font(29)
    font_axis = load_font(20)
    font_value = load_font(19)

    draw.rectangle([x0, y0, x0 + w, y0 + h], outline=(220, 220, 220), width=2)
    draw.text((x0 + 20, y0 + 15), title, fill=(25, 25, 25), font=font_title)

    ax_x0 = x0 + margin_left
    ax_y0 = y0 + margin_top
    ax_x1 = x0 + w - margin_right
    ax_y1 = y0 + h - margin_bottom
    draw.line([ax_x0, ax_y1, ax_x1, ax_y1], fill=(60, 60, 60), width=3)
    draw.line([ax_x0, ax_y1, ax_x0, ax_y0], fill=(60, 60, 60), width=3)

    vmax = max(values) * 1.1
    if vmax <= 0:
        vmax = 1.0

    ticks = 5
    for t in range(ticks + 1):
        ratio = t / ticks
        y = ax_y1 - (ax_y1 - ax_y0) * ratio
        v = vmax * ratio
        draw.line([ax_x0 - 8, y, ax_x0, y], fill=(80, 80, 80), width=2)
        draw.text((ax_x0 - 58, y - 9), f"{v:.1f}", fill=(90, 90, 90), font=font_axis)

    n = len(values)
    slot = (ax_x1 - ax_x0) / max(n, 1)
    bar_w = int(slot * 0.55)

    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        cx = ax_x0 + slot * (i + 0.5)
        left = int(cx - bar_w / 2)
        right = int(cx + bar_w / 2)
        top = int(ax_y1 - (ax_y1 - ax_y0) * (value / vmax))
        draw.rectangle([left, top, right, ax_y1], fill=color, outline=(40, 40, 40), width=2)

        vt = value_fmt.format(value)
        bbox = draw.textbbox((0, 0), vt, font=font_value)
        tw = bbox[2] - bbox[0]
        draw.text((int(cx - tw / 2), top - 24), vt, fill=(20, 20, 20), font=font_value)

        lb = draw.textbbox((0, 0), label, font=font_axis)
        lw = lb[2] - lb[0]
        draw.text((int(cx - lw / 2), ax_y1 + 10), label, fill=(20, 20, 20), font=font_axis)


def build_metric_figure() -> None:
    main_metrics = load_json(ROOT / "outputs" / "h100_main" / "eval" / "token_loss_metrics.json")
    ablation_metrics = load_json(
        ROOT / "outputs" / "h100_ablation_block16" / "eval" / "token_loss_metrics.json"
    )

    labels = ["Base", "OFT-32", "OFT-16"]
    nll_values = [
        float(main_metrics["base"]["mean_nll"]),
        float(main_metrics["oft"]["mean_nll"]),
        float(ablation_metrics["oft"]["mean_nll"]),
    ]
    ppl_values = [
        float(main_metrics["base"]["perplexity"]),
        float(main_metrics["oft"]["perplexity"]),
        float(ablation_metrics["oft"]["perplexity"]),
    ]

    colors = [(150, 150, 150), (49, 120, 198), (71, 159, 93)]

    canvas = Image.new("RGB", (1800, 740), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw_bar_panel(
        draw,
        title="Token-level Mean NLL (lower is better)",
        labels=labels,
        values=nll_values,
        colors=colors,
        panel=(40, 30, 860, 660),
        value_fmt="{:.4f}",
    )
    draw_bar_panel(
        draw,
        title="Perplexity (lower is better)",
        labels=labels,
        values=ppl_values,
        colors=colors,
        panel=(900, 30, 860, 660),
        value_fmt="{:.4f}",
    )

    canvas.save(FIG_DIR / "metric_bars.png")


def copy_existing_figures() -> None:
    pairs = {
        ROOT / "outputs" / "h100_main" / "eval" / "loss_curve.png": FIG_DIR / "main_loss_curve.png",
        ROOT / "outputs" / "h100_ablation_block16" / "eval" / "loss_curve.png": FIG_DIR / "ablation_loss_curve.png",
        ROOT / "artifacts" / "loss_comparison.png": FIG_DIR / "loss_comparison.png",
    }
    for src, dst in pairs.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source figure: {src}")
        shutil.copy2(src, dst)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    copy_existing_figures()
    build_metric_figure()
    print(f"Figures are ready in: {FIG_DIR}")


if __name__ == "__main__":
    main()
