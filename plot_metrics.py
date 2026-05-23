import os
import json
import argparse
import matplotlib.pyplot as plt

def plot_metrics(metrics_file, output_file):
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found at {metrics_file}. Please run training first.")
        
    with open(metrics_file, "r") as f:
        history = json.load(f)
        
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    train_acc = history.get("train_acc", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])
    
    if not epochs:
        print("No epochs found in metrics history.")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Loss Subplot
    ax1.plot(epochs, train_loss, label="Train Loss", color="royalblue", marker='o', linewidth=2)
    # Check if validation split was used (val_loss is not None/empty)
    if any(v is not None for v in val_loss):
        # Filter None values if any
        val_loss_filtered = [v for v in val_loss if v is not None]
        epochs_val = [epochs[i] for i, v in enumerate(val_loss) if v is not None]
        ax1.plot(epochs_val, val_loss_filtered, label="Val Loss", color="darkorange", marker='x', linewidth=2, linestyle='--')
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=11)
    
    # 2. Accuracy Subplot
    # Convert accuracy decimal values to percentages for plotting
    train_acc_pct = [a * 100 for a in train_acc]
    ax2.plot(epochs, train_acc_pct, label="Train Acc", color="forestgreen", marker='o', linewidth=2)
    if any(v is not None for v in val_acc):
        val_acc_pct = [v * 100 for v in val_acc if v is not None]
        epochs_val = [epochs[i] for i, v in enumerate(val_acc) if v is not None]
        ax2.plot(epochs_val, val_acc_pct, label="Val Acc", color="crimson", marker='x', linewidth=2, linestyle='--')
    ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Successfully saved metrics curves plot to {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Attention-LSTM Jazz Music Generator training history")
    parser.add_argument("--metrics_file", type=str, default="checkpoints/metrics.json", help="Path to metrics.json file")
    parser.add_argument("--output", type=str, default="metrics_plot.png", help="Path to save generated plot PNG file")
    
    args = parser.parse_args()
    
    plot_metrics(args.metrics_file, args.output)
