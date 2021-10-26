import argparse
from importlib import import_module

import cudf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find suitable checkpoints for testing"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Trained model directory",
        default="data/example_model",
    )
    parser.add_argument("--top_n", type=int, help="Number of checkpoints", default=10)
    args = parser.parse_args()

    model_dir_temp = args.model_dir
    if model_dir_temp.endswith("/"):
        model_dir_temp = model_dir_temp[:-1]

    # grab params from train_config
    cfg_file = "train_config"
    model_dir_string = model_dir_temp.replace("/", ".") + ".configs." + cfg_file
    model_arguments = import_module(model_dir_string)
    Config = getattr(model_arguments, "Config")
    train_cfg = Config()
    # load progress.csv
    model_csv = model_dir_temp + "/progress.csv"
    og_df = cudf.read_csv(model_csv)
    og_df = og_df.drop(["fps", "misc/total_timesteps"], axis=1)

    # grab selectable models & append last row as it will be some strange number
    col_name = "misc/nupdates"
    save_interval = train_cfg.training.save_interval
    df = og_df[og_df[col_name] % save_interval == 0].append(og_df.tail(1))

    # find top N models, with highest mean rewards
    n = args.top_n
    candidates = df.nlargest(n, ["eprewmean"])
    candidates = candidates.sort_values(by=["loss/policy_entropy"])
    print(candidates)
