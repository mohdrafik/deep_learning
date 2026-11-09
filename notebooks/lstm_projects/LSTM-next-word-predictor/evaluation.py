"""Training-performance reporting: perplexity and loss/accuracy curves."""

import matplotlib.pyplot as plt
import numpy as np


def perplexity(loss):
    return float(np.exp(loss))


def plot_history(history, save_path):
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, history_dict["loss"], label="train")
    if "val_loss" in history_dict:
        ax_loss.plot(epochs, history_dict["val_loss"], label="validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("sparse categorical crossentropy")
    ax_loss.legend()

    ax_acc.plot(epochs, history_dict["accuracy"], label="train")
    if "val_accuracy" in history_dict:
        ax_acc.plot(epochs, history_dict["val_accuracy"], label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def print_summary(history):
    history_dict = history.history
    final_train_loss = history_dict["loss"][-1]
    print(f"Final train loss: {final_train_loss:.4f} | perplexity: {perplexity(final_train_loss):.2f}")
    print(f"Final train accuracy: {history_dict['accuracy'][-1]:.4f}")

    if "val_loss" in history_dict:
        final_val_loss = history_dict["val_loss"][-1]
        print(f"Final val loss:   {final_val_loss:.4f} | perplexity: {perplexity(final_val_loss):.2f}")
        print(f"Final val accuracy:   {history_dict['val_accuracy'][-1]:.4f}")
