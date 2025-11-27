import matplotlib.pyplot as plt
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("to_display", choices=["bleu", "loss"])
    args = parser.parse_args()

    # Load data (ignore columns that aren't used)
    if args.to_display == "bleu":
        df = pd.read_csv(args.data_path, usecols=["batch", "train_bleu", "test_bleu"])
    else:
        df = pd.read_csv(args.data_path, usecols=["batch", "train_loss", "test_loss"])

    df.plot(x="batch")
    plt.xlabel("Batch")

    # Change title and y axis label depending on data
    if args.to_display == "bleu":
        plt.title("BLEU score over batches")
        plt.ylabel("BLEU score")
    else:
        df = pd.read_csv(args.data_path, usecols=["batch", "train_loss", "test_loss"])
        plt.title("Loss over batches")
        plt.ylabel("Loss (cross entropy)")

    plt.show()

if __name__ == "__main__":
    main()