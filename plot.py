import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from crowd_sim.envs.utils.helper import smooth_data
from matplotlib.widgets import Slider, Button


def main(save_figures=False):
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    # add more training curves by directory name here!
    models_list = ["fov_90", "fov_180", "fov_360"]
    model_dicts = {}
    for i in range(len(models_list)):
        model_csv = "data/" + models_list[i] + "/progress.csv"
        # key: model name, value: model csv DataFrame
        model_dicts[models_list[i]] = pd.read_csv(model_csv)

    metric_list = [
        "eprewmean",
        "loss/policy_entropy",
        "loss/policy_loss",
        "loss/value_loss",
    ]
    for i in range(len(metric_list)):
        fig, ax = plt.subplots()
        # fig.suptitle(metric_list[i])
        plt.title(metric_list[i])
        plt.grid()
        k = 0
        for key in model_dicts.keys():
            if metric_list[i] not in model_dicts[key].keys():
                continue
            else:
                x_axis = model_dicts[key]["misc/total_timesteps"].tolist()
                y_raw = model_dicts[key][metric_list[i]].tolist()
                if "entropy" in metric_list[i]:
                    ax.plot(x_axis, y_raw, colors[k], alpha=0.95, label=key)
                else:
                    y_smooth = smooth_data(y_raw, weight=0.95)
                    ax.plot(x_axis, y_raw, colors[k], alpha=0.15)
                    ax.plot(x_axis, y_smooth, colors[k], alpha=0.95, label=key)

                print(
                    "Model",
                    str(key),
                    "avg",
                    metric_list[i],
                    np.average(model_dicts[key][metric_list[i]]),
                )
                k += 1
        ax.set_xlabel("total_timesteps")
        ax.legend(loc="upper left")
        if save_figures:
            f_name = "metric" + str(i + 1) + ".jpg"
            plt.savefig(f_name)
            print("Saved to ", f_name)
        print("------------------------")
    plt.show()

if __name__ == '__main__':
    main(save_figures=True)
