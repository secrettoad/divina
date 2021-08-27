from unittest.mock import patch
import os
from ..cli.cli import (
    cli_build_dataset,
    cli_train_vision,
    cli_predict_vision,
    cli_validate_vision,
)
import dask.dataframe as ddf
import pandas as pd
import joblib
from ..utils import compare_sk_models
import json


@patch.dict(
    os.environ, {"DATASET_BUCKET": "{}/dataset".format(os.environ["TEST_BUCKET"])}
)
@patch.dict(os.environ, {"DATASET_ID": "test1"})
@patch.dict(os.environ, {"DATA_BUCKET": "{}/data".format(os.environ["TEST_BUCKET"])})
def test_build_dataset_small(s3_fs, test_df_1, dask_client_remote):
    test_df_1.to_csv(
        os.path.join(os.environ["DATA_BUCKET"], "test_df_1.csv"), index=False
    )
    cli_build_dataset(
        s3_fs=s3_fs,
        dataset_name=os.environ["DATASET_ID"],
        write_path=os.environ["DATASET_BUCKET"],
        read_path=os.environ["DATA_BUCKET"],
        ec2_keypair_name="divina2",
        dask_client=dask_client_remote,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(os.environ["DATASET_BUCKET"], os.environ["DATASET_ID"], "data")
        ).compute(),
        test_df_1,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                os.environ["DATASET_BUCKET"], os.environ["DATASET_ID"], "profile"
            )
        ).compute(),
        test_df_1.describe(),
    )


@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(
    os.environ, {"VISION_BUCKET": "{}/vision".format(os.environ["TEST_BUCKET"])}
)
def test_train_small(s3_fs, test_df_1, test_model_1, test_vd_3, dask_client_remote):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_vd_3["vision_definition"]["dataset_directory"],
            test_vd_3["vision_definition"]["dataset_id"],
            "data",
        )
    )
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(
            test_vd_3["vision_definition"]["dataset_directory"],
            test_vd_3["vision_definition"]["dataset_id"],
            "profile",
        )
    )
    vision_definition = test_vd_3["vision_definition"]
    cli_train_vision(
        s3_fs=s3_fs,
        vision_definition=vision_definition,
        write_path=os.environ["VISION_BUCKET"],
        vision_name=os.environ["VISION_ID"],
        keep_instances_alive=False,
        dask_client=dask_client_remote,
        ec2_keypair_name="divina2",
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
def test_predict_small(
    s3_fs, test_df_1, test_model_1, test_predictions_1, test_vd_3, dask_client_remote
):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_vd_3["vision_definition"]["dataset_directory"],
            test_vd_3["vision_definition"]["dataset_id"],
            "data",
        )
    )
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(
            test_vd_3["vision_definition"]["dataset_directory"],
            test_vd_3["vision_definition"]["dataset_id"],
            "profile",
        )
    )
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
    vision_definition = test_vd_3["vision_definition"]
    cli_predict_vision(
        s3_fs=s3_fs,
        vision_definition=vision_definition,
        write_path=os.environ["VISION_BUCKET"],
        read_path=os.environ["VISION_BUCKET"],
        vision_name=os.environ["VISION_ID"],
        keep_instances_alive=False,
        dask_client=dask_client_remote,
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
def test_validate_small(
    s3_fs, test_vd_3, test_df_1, test_metrics_1, test_predictions_1, dask_client_remote
):
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_vd_3["vision_definition"]["dataset_directory"],
            test_vd_3["vision_definition"]["dataset_id"],
            "data",
        )
    )
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(
            test_vd_3["vision_definition"]["dataset_directory"],
            test_vd_3["vision_definition"]["dataset_id"],
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

    cli_validate_vision(
        s3_fs=s3_fs,
        vision_definition=test_vd_3["vision_definition"],
        write_path=os.environ["VISION_BUCKET"],
        read_path=os.environ["VISION_BUCKET"],
        vision_name=os.environ["VISION_ID"],
        ec2_keypair_name="divina2",
        keep_instances_alive=False,
        local=False,
        dask_client=dask_client_remote,
    )

    with s3_fs.open(
        os.path.join(
            os.environ["VISION_BUCKET"], os.environ["VISION_ID"], "metrics.json"
        ),
        "r",
    ) as f:
        metrics = json.load(f)

    assert metrics == test_metrics_1
