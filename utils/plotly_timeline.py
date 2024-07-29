import plotly.express as px
import pandas as pd
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark timeline plotter")
    parser.add_argument(
        "log_file_path",
        type=str,
        help="path to the log file of the benchmark",
    )
    return parser.parse_args()


def merge_start_end(df):
    stack = dict()
    rows = []

    for idx in tqdm(range(1, df.shape[0])):
        next_row = df.iloc[idx]
        if len(stack.get(next_row["pipeline_name"], [])) == 0:
            stack[next_row["pipeline_name"]] = [next_row]
            continue
        if next_row[["pipeline_name", "module_name", "phase"]].equals(
            stack[next_row["pipeline_name"]][-1][
                ["pipeline_name", "module_name", "phase"]
            ]
        ):
            matched_row = stack[next_row["pipeline_name"]].pop()
            rows.append(
                {
                    "Start timestamp": matched_row["timestamp"],
                    "End timestamp": next_row["timestamp"],
                    "Pipeline name": matched_row["pipeline_name"].strip(),
                    "Module name": matched_row["module_name"].strip(),
                    "Phase": matched_row["phase"].strip(),
                    "Level": (
                        "Level 0"
                        if matched_row["module_name"].strip().find("pipeline") == 0
                        else "Level 1"
                    ),
                    "Query submitted": matched_row["submitted"],
                    "Epoch": matched_row["epoch"],
                    "Batch": matched_row["batch"],
                }
            )
        else:
            stack[next_row["pipeline_name"]].append(next_row)

    new_df = pd.DataFrame.from_dict(rows, orient="columns")
    return new_df


def merge_details_from_pipeline(df):
    pipeline = df[df["Level"] == "Level 0"]

    for idx in range(len(pipeline)):
        idx_whole = df[
            (df["Start timestamp"] > pipeline.iloc[idx]["Start timestamp"])
            & (df["End timestamp"] < pipeline.iloc[idx]["End timestamp"])
            & (df["Pipeline name"] == pipeline.iloc[idx]["Pipeline name"])
        ].index

        df.loc[idx_whole, "Batch"] = pipeline.iloc[idx]["Batch"]
        df.loc[idx_whole, "Epoch"] = pipeline.iloc[idx]["Epoch"]

    return df


def main():
    args = parse_args()
    df = pd.read_csv(
        args.log_file_path,
        names=[
            "timestamp",
            "pipeline_name",
            "module_name",
            "phase",
            "state",
            "submitted",
            "epoch",
            "batch",
        ],
    )
    df = df.dropna(thresh=4)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["submitted"] = pd.to_datetime(df["submitted"], unit="s")

    # merge start and end records of events into one
    new_df = merge_start_end(df)
    new_df = merge_details_from_pipeline(new_df)

    # calculate the timedelta between start and end of an event in milliseconds
    new_df["Duration"] = (
        new_df["End timestamp"] - new_df["Start timestamp"]
    ) / pd.Timedelta(milliseconds=1)
    new_df["Slack"] = (
        new_df["Start timestamp"] - new_df["Query submitted"]
    ) / pd.Timedelta(milliseconds=1)

    new_df["Slack"] = new_df["Slack"].fillna("n/a")

    # create a map of columns to show in a tooltip
    hover_data = {
        "Duration": True,
        "Module name": False,
        "Start timestamp": True,
        "End timestamp": True,
        "Pipeline name": False,
        "Level": False,
        "Phase": True,
        "Query submitted": False,
        "Slack": True,
        "Epoch": True,
        "Batch": True,
    }

    fig = px.timeline(
        new_df,
        x_start="Start timestamp",
        x_end="End timestamp",
        y="Level",
        color="Module name",
        hover_data=hover_data,
        hover_name="Module name",
        facet_row="Pipeline name",
    )
    fig.update_yaxes(
        autorange="reversed"
    )  # otherwise levels are listed from the top down

    fig.show()


if __name__ == "__main__":
    main()
