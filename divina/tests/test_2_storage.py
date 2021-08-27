from unittest.mock import patch
import os
from ..dataset import build_dataset_dask
from dask import dataframe as ddf
import pandas as pd
import json
from ..utils import compare_sk_models
from dask_ml.linear_model import LinearRegression
from ..train import dask_train
import joblib
from ..predict import dask_predict
from ..validate import dask_validate


@patch.dict(
    os.environ, {"DATASET_BUCKET": "{}/dataset".format(os.environ["TEST_BUCKET"])}
)
@patch.dict(os.environ, {"DATASET_ID": "test1"})
@patch.dict(os.environ, {"DATA_BUCKET": "{}/data".format(os.environ["TEST_BUCKET"])})
def test_build_dataset_small(s3_fs, test_df_1):
    test_df_1.to_csv(
        os.path.join(os.environ["DATA_BUCKET"], "test_df_1.csv"), index=False
    )
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join("stubs", os.environ["DATASET_ID"], "data")
    )
    ddf.from_pandas(test_df_1, chunksize=10000).describe().to_parquet(
        os.path.join("stubs", os.environ["DATASET_ID"], "profile")
    )
    build_dataset_dask(
        s3_fs=s3_fs,
        dataset_name=os.environ["DATASET_ID"],
        write_path=os.environ["DATASET_BUCKET"],
        read_path=os.environ["DATA_BUCKET"],
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(os.environ["DATASET_BUCKET"], os.environ["DATASET_ID"], "data")
        ).compute(),
        ddf.read_parquet(
            os.path.join("stubs", os.environ["DATASET_ID"], "data")
        ).compute(),
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                os.environ["DATASET_BUCKET"], os.environ["DATASET_ID"], "profile"
            )
        ).compute(),
        ddf.read_parquet(
            os.path.join("stubs", os.environ["DATASET_ID"], "profile")
        ).compute(),
    )


@patch.dict(
    os.environ, {"VISION_BUCKET": "{}/vision".format(os.environ["TEST_BUCKET"])}
)
@patch.dict(os.environ, {"VISION_ID": "test1"})
def test_train(s3_fs, test_df_1, test_vd_1, dask_client, test_model_1):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_vd_1["vision_definition"]["dataset_directory"],
            test_vd_1["vision_definition"]["dataset_id"],
            "data",
        )
    )
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(
            test_vd_1["vision_definition"]["dataset_directory"],
            test_vd_1["vision_definition"]["dataset_id"],
            "profile",
        )
    )
    with open(
        os.path.join(
            test_vd_1["vision_definition"]["dataset_directory"],
            test_vd_1["vision_definition"]["dataset_id"],
            "vision_definition.json",
        ),
        "w+",
    ) as f:
        json.dump(test_vd_1, f)
    dask_train(
        s3_fs=s3_fs,
        dask_model=LinearRegression,
        vision_definition=test_vd_1["vision_definition"],
        vision_id=os.environ["VISION_ID"],
        write_path=os.environ["VISION_BUCKET"],
    )

    with s3_fs.open(
        os.path.join(
            os.environ["VISION_BUCKET"],
            os.environ["VISION_ID"],
            "models",
            "s-19700101-000008_h-1",
        ),
        "rb",
    ) as f:
        assert compare_sk_models(joblib.load(f), test_model_1)


@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(
    os.environ, {"VISION_BUCKET": "{}/vision".format(os.environ["TEST_BUCKET"])}
)
def test_predict(
    s3_fs, dask_client, test_df_1, test_vd_1, test_predictions_1, test_model_1
):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_vd_1["vision_definition"]["dataset_directory"],
            test_vd_1["vision_definition"]["dataset_id"],
            "data",
        )
    )
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(
            test_vd_1["vision_definition"]["dataset_directory"],
            test_vd_1["vision_definition"]["dataset_id"],
            "profile",
        )
    )
    joblib.dump(test_model_1, "tmp")
    joblib.dump(test_model_1, "s-19700101-000008_h-1")
    s3_fs.put(
        "s-19700101-000008_h-1",
        os.path.join(
            os.environ["VISION_BUCKET"],
            os.environ["VISION_ID"],
            "models",
            "s-19700101-000008_h-1",
        ),
        recursive=True,
    )
    os.remove("s-19700101-000008_h-1")

    dask_predict(
        s3_fs=s3_fs,
        vision_definition=test_vd_1["vision_definition"],
        vision_id=os.environ["VISION_ID"],
        read_path=os.environ["VISION_BUCKET"],
        write_path=os.environ["VISION_BUCKET"],
    )

    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                os.environ["VISION_BUCKET"],
                os.environ["VISION_ID"],
                "predictions",
                "s-19700101-000008",
            )
        ).compute(),
        test_predictions_1,
    )


@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(
    os.environ, {"VISION_BUCKET": "{}/vision".format(os.environ["TEST_BUCKET"])}
)
def test_validate(
    s3_fs, test_vd_1, test_df_1, test_metrics_1, test_predictions_1, dask_client
):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_vd_1["vision_definition"]["dataset_directory"],
            test_vd_1["vision_definition"]["dataset_id"],
            "data",
        )
    )
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(
            test_vd_1["vision_definition"]["dataset_directory"],
            test_vd_1["vision_definition"]["dataset_id"],
            "profile",
        )
    )

    ddf.from_pandas(test_predictions_1, chunksize=10000).to_parquet(
        os.path.join(
            os.environ["VISION_BUCKET"],
            os.environ["VISION_ID"],
            "predictions",
            "s-19700101-000008",
        )
    )

    dask_validate(
        s3_fs=s3_fs,
        vision_definition=test_vd_1["vision_definition"],
        vision_id=os.environ["VISION_ID"],
        read_path=os.environ["VISION_BUCKET"],
        write_path=os.environ["VISION_BUCKET"],
    )

    with s3_fs.open(
        os.path.join(
            os.environ["VISION_BUCKET"], os.environ["VISION_ID"], "metrics.json"
        ),
        "r",
    ) as f:
        metrics = json.load(f)

    assert metrics == test_metrics_1
