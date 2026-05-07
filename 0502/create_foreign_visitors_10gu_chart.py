import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib import font_manager, rcParams


HERE = Path(__file__).resolve().parent
CSV_PATH = next(HERE.glob("*.csv"))
OUT_PATH = HERE / "foreign_visitors_10gu_yearly_average_line.png"

font_candidates = ["Malgun Gothic", "NanumGothic", "Arial Unicode MS"]
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
for font in font_candidates:
    if font in available_fonts:
        rcParams["font.family"] = font
        break
rcParams["axes.unicode_minus"] = False

our_gu = [
    "\uac15\ub0a8\uad6c",
    "\uac15\uc11c\uad6c",
    "\ub9c8\ud3ec\uad6c",
    "\uc11c\ucd08\uad6c",
    "\uc131\ub3d9\uad6c",
    "\uc1a1\ud30c\uad6c",
    "\uc601\ub4f1\ud3ec\uad6c",
    "\uc6a9\uc0b0\uad6c",
    "\uc885\ub85c\uad6c",
    "\uc911\uad6c",
]

df = pd.read_csv(CSV_PATH)
df["\ub0a0\uc9dc"] = pd.to_numeric(df["\ub0a0\uc9dc"], errors="coerce").astype("Int64")
df["\uc678\uad6d\uc778 \ubc29\ubb38\uc790\uc218"] = pd.to_numeric(
    df["\uc678\uad6d\uc778 \ubc29\ubb38\uc790\uc218"], errors="coerce"
)

plot_df = (
    df[df["\uc9c0\uc5ed"].isin(our_gu)]
    .groupby(["\ub0a0\uc9dc", "\uc9c0\uc5ed"], as_index=False)["\uc678\uad6d\uc778 \ubc29\ubb38\uc790\uc218"]
    .mean()
    .pivot(index="\ub0a0\uc9dc", columns="\uc9c0\uc5ed", values="\uc678\uad6d\uc778 \ubc29\ubb38\uc790\uc218")
    .reindex(columns=our_gu)
    .sort_index()
)

years = plot_df.index.astype(int).tolist()
legend_order = plot_df.loc[2025].sort_values(ascending=False).index.tolist()
colors = {
    "\uac15\ub0a8\uad6c": "#ff7f50",
    "\uac15\uc11c\uad6c": "#9bd35a",
    "\ub9c8\ud3ec\uad6c": "#f58bdc",
    "\uc11c\ucd08\uad6c": "#ff6b7e",
    "\uc131\ub3d9\uad6c": "#f5a85a",
    "\uc1a1\ud30c\uad6c": "#2f80ed",
    "\uc601\ub4f1\ud3ec\uad6c": "#02c875",
    "\uc6a9\uc0b0\uad6c": "#ff7fa3",
    "\uc885\ub85c\uad6c": "#50f050",
    "\uc911\uad6c": "#ffc247",
}
markers = ["*", "s", "h", "X", "P", "^", "o", "8", "<", "D"]

fig, ax = plt.subplots(figsize=(15.5, 6.4), dpi=160)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for index, gu in enumerate(our_gu):
    y = plot_df[gu]
    line, = ax.plot(
        years,
        y,
        label=gu,
        color=colors[gu],
        linewidth=2.0,
        marker=markers[index % len(markers)],
        markersize=10.5,
        markeredgewidth=0,
        zorder=3,
    )
    line.set_path_effects(
        [pe.SimpleLineShadow(offset=(1.8, -1.8), alpha=0.22, rho=0.95), pe.Normal()]
    )
    ax.scatter(
        years,
        y,
        s=118,
        color=colors[gu],
        marker=markers[index % len(markers)],
        edgecolors="none",
        zorder=4,
    )

ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
ax.set_ylim(0, max(plot_df.max()) * 1.15)
ax.set_xticks(years)
ax.tick_params(axis="x", labelrotation=35, labelsize=10)
ax.tick_params(axis="y", labelsize=10)
ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, pos: "0" if x == 0 else f"{int(x / 1000):,} k")
)

ax.grid(True, which="major", color="#9a9a9a", alpha=0.32, linewidth=0.85)
for spine in ax.spines.values():
    spine.set_color("#d0d0d0")
    spine.set_linewidth(0.9)

ax.set_title(
    "\uc678\uad6d\uc778 \uc9c0\uc5ed\ubcc4 \ubc29\ubb38\uc790 \uc218 \ucd94\uc774 - 10\uac1c\uad6c \uc5f0\ub3c4\ubcc4 \ud3c9\uade0",
    fontsize=16,
    pad=18,
    weight="bold",
)
ax.set_xlabel("")
ax.set_ylabel("")

handles, labels = ax.get_legend_handles_labels()
handle_by_label = dict(zip(labels, handles))
ax.legend(
    [handle_by_label[label] for label in legend_order],
    legend_order,
    title="2025년 기준",
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    ncol=1,
    frameon=False,
    fontsize=10.5,
    title_fontsize=11,
    handlelength=2.2,
    handletextpad=0.5,
)

plt.subplots_adjust(left=0.07, right=0.84, top=0.86, bottom=0.13)
fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")

print(os.fspath(OUT_PATH))
print(plot_df.round(2).to_string())
