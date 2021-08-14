import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crowd_sim.envs.utils.helper import smooth_data


def create_mpl_label(model_name):
    if "fov" in model_name:
        label_str = model_name.replace("_", "-").replace("f", "F").replace("v", "V")
    elif "group" in model_name:
        label_str = model_name.replace("_env_", "-").replace("g", "G")
    return label_str


def main(save_figures=False, show_checkpoints=False):
    COLORS = ("b", "g", "r", "c", "m", "y", "k", "w")
    CHECKPOINT_COLOR = "#00FFFF"  # cyan

    # add more training curves by directory name here!
    models_list = ["example_model"]
    checkpoint_list = []
    # models_list = ["group_env_10", "group_env_15", "group_env_20"]
    # checkpoint_list = [9864360, 9792360, 9993960]
    # models_list = ["fov_90", "fov_180", "fov_360"]
    # checkpoint_list = [9000360, 8784360, 9360360]

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
        smoothing = 0.95
        for key in model_dicts.keys():
            if metric_list[i] not in model_dicts[key].keys():
                continue
            else:
                x_axis = model_dicts[key]["misc/total_timesteps"].tolist()
                y_raw = model_dicts[key][metric_list[i]].tolist()
                label_str = create_mpl_label(key)

                if "entropy" in metric_list[i]:
                    ax.plot(
                        x_axis, y_raw, COLORS[k], alpha=0.95, label=label_str + " Raw"
                    )
                else:
                    y_smooth = smooth_data(y_raw, weight=smoothing)
                    ax.plot(
                        x_axis, y_raw, COLORS[k], alpha=0.15, label=label_str + " Raw"
                    )
                    ax.plot(
                        x_axis,
                        y_smooth,
                        COLORS[k],
                        alpha=0.95,
                        label=label_str + " Smooth",
                    )
                if show_checkpoints:
                    index = x_axis.index(checkpoint_list[k])
                    ax.plot(
                        checkpoint_list[k],
                        y_raw[index],
                        COLORS[k],
                        marker=".",
                        markersize=6,
                    )

                print(
                    "Model",
                    str(key),
                    "avg",
                    metric_list[i],
                    np.average(model_dicts[key][metric_list[i]]),
                )
                k += 1
        fig.set_figheight(5)
        fig.set_figwidth(8)
        ax.set_xlabel("total_timesteps")
        # shrink axes by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if save_figures:
            f_name = "metric" + str(i + 1) + ".jpg"
            plt.savefig(f_name)
            print("Saved to ", f_name)
        print("------------------------")
    plt.show()


if __name__ == "__main__":
    main(False, False)
    # main(False, True)
    # main(True, True)
