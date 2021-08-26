import sys
import os
import dask.dataframe as dd
import joblib
import pandas as pd
from .dataset import get_dataset
import pathlib
import backoff
from botocore.exceptions import ClientError


@backoff.on_exception(backoff.expo, ClientError, max_time=30)
def dask_train(s3_fs, dask_model, vision_definition, write_path, vision_id):
    if write_path[:5] == "s3://":
        if not s3_fs.exists(write_path):
            s3_fs.mkdir(
                "{}/{}".format(write_path, "models"),
                create_parents=True,
                region_name=os.environ["AWS_DEFAULT_REGION"],
                acl="private",
            )
    else:
        pathlib.Path(os.path.join(write_path, vision_id), "models").mkdir(
            parents=True, exist_ok=True
        )
    df, profile = get_dataset(vision_definition)

    sys.stdout.write("Loading dataset\n")

    df[vision_definition["time_index"]] = dd.to_datetime(
        df[vision_definition["time_index"]], unit="s"
    )

    if "drop_features" in vision_definition:
        df = df.drop(columns=vision_definition["drop_features"])

    for h in vision_definition["time_horizons"]:
        if "signal_dimension" in vision_definition:
            df["{}_h_{}".format(vision_definition["target"], h)] = df.groupby(
                vision_definition["signal_dimensions"]
            )[vision_definition["target"]].shift(-h)
        else:
            df["{}_h_{}".format(vision_definition["target"], h)] = df[
                vision_definition["target"]
            ].shift(-h)

    models = {}

    time_min, time_max = (
        df[vision_definition["time_index"]].min().compute(),
        df[vision_definition["time_index"]].max().compute(),
    )

    for s in vision_definition["time_validation_splits"]:
        if pd.to_datetime(str(s)) <= time_min or pd.to_datetime(str(s)) >= time_max:
            raise Exception("Bad Time Split: {} | Check Dataset Time Range".format(s))
        df_train = df[df[vision_definition["time_index"]] < s]
        for h in vision_definition["time_horizons"]:

            model = dask_model()

            model.fit(
                df_train[
                    [
                        c
                        for c in df_train.columns
                        if not c
                        in [
                            "{}_h_{}".format(vision_definition["target"], h)
                            for h in vision_definition["time_horizons"]
                        ]
                        + [vision_definition["time_index"], vision_definition["target"]]
                    ]
                ].to_dask_array(lengths=True),
                df_train["{}_h_{}".format(vision_definition["target"], h)],
            )

            sys.stdout.write("Pipeline fit for horizon {}\n".format(h))

            models[
                "s-{}_h-{}".format(pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"), h)
            ] = model

            with s3_fs.open(
                "{}/{}/models/s-{}_h-{}".format(
                    write_path,
                    vision_id,
                    pd.to_datetime(str(s)).strftime("%Y%m%d-%H%M%S"),
                    h,
                ),
                "wb",
            ) as f:
                joblib.dump(model, f)

            sys.stdout.write("Pipeline persisted for horizon {}\n".format(h))

    return models
