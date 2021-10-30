import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crowd_sim.envs.utils.helper import smooth_data


def create_mpl_label(str_):
    # if str_ == "fov_360_reward_norm":
    #     str_ = "Potential-Based, Normalized"
    # elif str_ == "fov_360_reward_norm_exp_term_2":
    #     str_ = r"Shaped Reward, $\alpha$ = 2, Normalized"
    # elif str_ == "fov_360_reward_norm_exp_term_6":
    #     str_ = r"Shaped Reward, $\alpha$ = 6, Normalized"

    # jank formatting
    str_ = str_.replace("_", "-")
    if "f" in str_.lower():
        str_ = str_.replace("f", "F").replace("v", "V")
    elif "g" in str_.lower():
        # Group Env processing
        str_ = str_.replace("g", "G").replace("env-", "")
    return str_


def format_metric_label(metric_str):
    if "eprew" in metric_str.lower():
        metric_str = "Mean Reward of Last 100 Episodes"
    elif "loss" in metric_str.lower():
        metric_str = metric_str.replace("loss/", "").replace("_", " ")
    return metric_str.title()  # title case


def main(save_figures=False, show_checkpoints=False):
    COLORS = ("b", "g", "r", "c", "m", "y", "k", "w")
    CHECKPOINT_COLOR = "#00FFFF"

    # add more training curves by directory name here!
    # Group environment
    # models_list = ["group_env_10", "group_env_15", "group_env_20"]
    # checkpoint_list = [8712360, 9792360, 9993960]
    # FoV environment
    # models_list = ["fov_90", "fov_180", "fov_360"]
    # checkpoint_list = [9000360, 8784360, 9288360]
    models_list = ['example_model']
    checkpoint_list = []

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
        plt.title(format_metric_label(metric_list[i]))
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
                        x_axis, y_raw, COLORS[k], alpha=0.8, label=label_str + " Raw"
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
                        alpha=1,
                        marker="x",
                        markersize=10,
                        label=label_str + " Selected",
                    )

                print(
                    "Model",
                    str(key),
                    "avg",
                    metric_list[i],
                    np.average(model_dicts[key][metric_list[i]]),
                )
                k += 1
        fig.set_figheight(6)
        fig.set_figwidth(12)
        ax.set_xlabel("Number of Time Steps")
        # shrink axes by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # ax.legend(loc="best")

        if save_figures:
            f_name = "metric" + str(i + 1) + ".jpg"
            plt.savefig(f_name)
            print("Saved to ", f_name)
        print("------------------------")
    plt.show()


if __name__ == "__main__":
    main(0, 0)
