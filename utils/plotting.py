import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def plot_graphs(data_list, titles, save_path=None, save_freq=1):
    plots = len(data_list)
    num_data_points = len(data_list[0])
    epochs = np.arange(0, num_data_points * save_freq, save_freq)
    assert len(titles) == plots, "Number of titles must match number of plots"
    assert all(len(data) == num_data_points for data in data_list), "All data lists must have the same length"
    fig, axes = plt.subplots(1, plots, figsize=(5*plots+2, 5))
    if plots == 1:
        axes = [axes]  # Ensure axes is iterable
    for ax, data, title in zip(axes, data_list, titles):
        ax.plot(epochs, data, color="blue")
        ax.set_title(title)
        ax.set_xlabel("Epochs")

        # Limit to at most 10 ticks
        if len(epochs) > 10:
            xticks = epochs[::len(epochs) // 10]
            if epochs[-1] not in xticks:
                xticks = np.append(xticks, epochs[-1])
        else:
            xticks = epochs
        ax.set_xticks(xticks)
    
    # Save or show
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
