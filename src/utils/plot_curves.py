import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_losses(csv_path, output_path=None):
    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(csv_path)

    required_cols = {"epoch", "train_loss"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain at least the following columns: {required_cols}"
        )

    epoch = df["epoch"]
    loss_columns = [col for col in df.columns if col != "epoch"]

    # ----------------------------
    # Matplotlib configuration (LaTeX-friendly)
    # ----------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 8,
        "legend.fontsize": 5,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    # Typical single-column width (~8.5 cm)
    fig, ax = plt.subplots(figsize=(3.35, 2.6))

    # ----------------------------
    # Plot curves
    # ----------------------------
    for col in loss_columns:
        if col in ("train_loss", "val_l1"):
            continue

        if col in ("val_l1_mel", "val_waveform"):
            ax.plot(
                epoch,
                df[col] * 10.0,
                linewidth=1.8,
                label=f"{col.replace('_', ' ').replace('val', '')} (x10)"
            )
        else:
            ax.plot(
                epoch,
                df[col],
                linewidth=1.8,
                label=col.replace('_', ' ').replace('val', '')
            )

    # ----------------------------
    # Axes, grid, legend
    # ----------------------------
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss value")

    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout(pad=0.5)

    # ----------------------------
    # Save / show
    # ----------------------------
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot training and validation loss curves from a CSV log file."
    )
    parser.add_argument(
        "csv",
        type=str,
        help="Path to the CSV log file containing training and validation losses."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output figure (PDF/PNG recommended)."
    )

    args = parser.parse_args()
    plot_losses(args.csv, args.output)


if __name__ == "__main__":
    main()