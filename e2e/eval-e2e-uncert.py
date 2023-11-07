from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["science", "ieee", "grid"])
plt.rcParams.update({"font.size": 12})


def plot_outline_with_uncertainty_envelope_and_torch(mean_outline, std_outline, xts, zs):
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(18 * cm, 9 * cm), dpi=600)

    xa = np.arange(0, len(mean_outline)) / 10
    ax.plot(xa, mean_outline, color="blue", alpha=1, ls="-", linewidth=1, label="mean prediction")

    ax.fill_between(
        xa,
        mean_outline,
        color="white",
        hatch="",
        edgecolor="black",
        alpha=0.8,
    )

    ax.fill_between(
        xa,
        mean_outline + std_outline,
        mean_outline - std_outline,
        color="red",
        alpha=0.3,
        label="$\pm 1\sigma$",
    )

    avg_layer_height = mean_outline.max() / zs.max() / 1.1
    xoffset = int(np.abs(np.argmax(mean_outline) - np.median(xts)))
    ax.scatter(
        (xts - xoffset) / 10,
        (zs + 1) * avg_layer_height - 1.5,
        marker="v",
        label="torch position",
        alpha=1,
        linewidths=0.75,
        s=15,
        facecolors="none",
        edgecolors="black",
    )

    ax.legend(
        # handles,
        # labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.45),
        frameon=False,
        fancybox=False,
        shadow=False,
    )
    ax.set_ylim(0, mean_outline.max() + 2)
    ax.set_xlim(0, xa.max())
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    fig.subplots_adjust(bottom=0.15)
    return fig


def main():
    sim_results = Path("./outputs/sim")

    mean_outline = np.load(sim_results / "mean_outline.npy")
    std_outline = np.load(sim_results / "std_outline.npy")

    xts = []
    zs = []

    start_idx = 279
    hatch_distance = 52
    offset = hatch_distance // 2

    L = 10
    for l in range(L + 1):
        zs.extend(np.array([l] * (L - l)))
        layer = np.arange(start_idx, start_idx + hatch_distance * (L - l), hatch_distance, dtype=int)
        xts.extend(layer)
        start_idx += offset

    xts = np.array(xts)
    zs = np.array(zs)

    fig = plot_outline_with_uncertainty_envelope_and_torch(mean_outline, std_outline, xts, zs)
    plt.show()

    sim = Path("/Users/magnus/Desktop/graphics/results/e2esim").absolute()
    fig.savefig(sim / "e2e_h-52_off-26_uncertainty_envelope.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
