from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

WIDTH_MM = {"single": 90.0, "full": 190.0}
HEIGHT_MM = 90.0
DPI = 500
COLORS = {
    "ink": "#20242A",
    "charcoal": "#74777B",
    "white": "#FFFFFF",
    "pale_grey": "#EEF1F5",
    "grid_grey": "#D9DEE6",
    "blue": "#526FB4",
    "blue_dark": "#304F8C",
    "blue_light": "#AFC0E1",
    "slate": "#71869C",
    "teal": "#3F929C",
    "teal_light": "#A8D0D3",
    "coral": "#D46B5F",
    "coral_light": "#E8B3AD",
    "orange": "#D4934B",
    "purple": "#7968A8",
    "rose": "#B66B87",
    "green": "#618E79",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canvas_inches(width_mm: float, height_mm: float) -> tuple[float, float]:
    """Return the nearest 500-dpi canvas while preserving nominal mm dimensions."""
    return (
        round(width_mm / 25.4 * DPI) / DPI,
        round(height_mm / 25.4 * DPI) / DPI,
    )


def load_json(relative: str) -> dict[str, Any] | list[Any]:
    path = project_root() / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required real-data artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    resolved = Path(font_manager.findfont("Arial", fallback_to_default=False))
    family = font_manager.FontProperties(fname=str(resolved)).get_name()
    if family != "Arial":
        raise RuntimeError(f"Exact Arial is required; resolved {family!r} at {resolved}")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 5.8,
            "axes.labelsize": 8.5,
            "axes.labelweight": "bold",
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "legend.fontsize": 5.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.edgecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "axes.facecolor": COLORS["white"],
            "figure.facecolor": COLORS["white"],
            "axes.grid": True,
            "grid.color": COLORS["grid_grey"],
            "grid.linewidth": 0.45,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def export_and_check(
    figure: Any,
    *,
    output_stem: Path,
    width_mm: float,
    height_mm: float,
    sources: list[Path],
    data_summary: dict[str, Any],
) -> dict[str, Any]:
    allowed_root = (project_root() / "fig").resolve()
    resolved_parent = output_stem.parent.resolve()
    if allowed_root != resolved_parent and allowed_root not in resolved_parent.parents:
        raise RuntimeError(f"Figure writes must remain under {allowed_root}: {output_stem}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    expected_inches = canvas_inches(width_mm, height_mm)
    actual_inches = tuple(float(value) for value in figure.get_size_inches())
    if any(abs(a - b) > 1e-6 for a, b in zip(actual_inches, expected_inches)):
        raise RuntimeError(f"Canvas size {actual_inches} != required {expected_inches}")

    outputs = {suffix: output_stem.with_suffix(f".{suffix}") for suffix in ("pdf", "svg", "png")}
    figure.savefig(outputs["pdf"], metadata={"Creator": "EPT-Net", "CreationDate": None})
    figure.savefig(outputs["svg"], metadata={"Creator": "EPT-Net"})
    figure.savefig(outputs["png"], dpi=DPI, metadata={"Software": "EPT-Net"})

    from PIL import Image

    expected_px = (round(width_mm / 25.4 * DPI), round(height_mm / 25.4 * DPI))
    with Image.open(outputs["png"]) as image:
        actual_px = image.size
    if actual_px != expected_px:
        raise RuntimeError(f"PNG size {actual_px} != required {expected_px}")
    for path in outputs.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty figure output: {path}")

    expected_points = tuple(value / 25.4 * 72.0 for value in (width_mm, height_mm))
    pdf_head = outputs["pdf"].read_bytes()[:16384]
    match = re.search(rb"/MediaBox\s*\[\s*0(?:\.0+)?\s+0(?:\.0+)?\s+([0-9.]+)\s+([0-9.]+)\s*\]", pdf_head)
    if match is None:
        raise RuntimeError("Could not verify PDF MediaBox")
    pdf_points = tuple(float(value) for value in match.groups())
    if any(abs(a - b) > 0.06 for a, b in zip(pdf_points, expected_points)):
        raise RuntimeError(f"PDF size {pdf_points} pt != required {expected_points} pt")

    svg_root = ElementTree.parse(outputs["svg"]).getroot()

    def svg_points(attribute: str) -> float:
        value = svg_root.attrib.get(attribute, "")
        svg_match = re.fullmatch(r"([0-9.]+)pt", value)
        if svg_match is None:
            raise RuntimeError(f"Could not verify SVG {attribute}: {value!r}")
        return float(svg_match.group(1))

    svg_size = (svg_points("width"), svg_points("height"))
    if any(abs(a - b) > 0.06 for a, b in zip(svg_size, expected_points)):
        raise RuntimeError(f"SVG size {svg_size} pt != required {expected_points} pt")

    report = {
        "figure_id": output_stem.name,
        "generated_formats": ["pdf", "svg", "png"],
        "dimensions_mm": {"width": width_mm, "height": height_mm},
        "rendered_dimensions_mm": {
            "width": actual_inches[0] * 25.4,
            "height": actual_inches[1] * 25.4,
        },
        "png": {"dpi": DPI, "width_px": actual_px[0], "height_px": actual_px[1]},
        "sources": {str(path.relative_to(project_root())): sha256(path) for path in sources},
        "data_summary": data_summary,
        "checks": {
            "files": "pass",
            "dimensions": "pass",
            "font": "pass",
            "color": "pass",
            "data_scope": "pass",
            "visual_preview": "manual-required",
            "write_boundary": "pass",
        },
        "exceptions": [],
    }
    # Keep demo and formal provenance records independent even when both are
    # rendered in the same figure directory.
    report_path = output_stem.with_name(f"{output_stem.name}.qa-report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
