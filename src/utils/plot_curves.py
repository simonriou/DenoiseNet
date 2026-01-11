import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_losses(csv_path, output_path=None):
    df = pd.read_csv(csv_path)

    required_cols = {"epoch", "train_loss"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain at least the following columns: {required_cols}"
        )
    
    epoch = df["epoch"]

    loss_columns = [col for col in df.columns if col not in ("epoch",)]

    plt.figure(figsize=(8, 5))


    for col in loss_columns:
        if col == "train_loss":
            continue
        elif col == "val_l1_mel" or col == "val_waveform":
            plt.plot(
                epoch,
                df[col] * 10,
                linewidth=2.0,
                linestyle="--",
                alpha=0.7,
                label=f"Validation - {col} (x10)"
            )
        else:
            plt.plot(
                epoch,
                df[col],
                linewidth=2.0,
                linestyle="--",
                alpha=0.7,
                label=f"Validation - {col}"
            )

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


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
        help="Path to save the output plot image. If not provided, the plot will be displayed on screen."
    )

    args = parser.parse_args()
    plot_losses(args.csv, args.output)

if __name__ == "__main__":
    main()