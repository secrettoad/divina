import os
import json
from unittest.mock import patch, MagicMock
from ..vision import validate_vision_definition
from ..train import dask_train
from ..predict import dask_predict
from ..dataset import get_dataset, build_dataset_dask
from ..validate import dask_validate
from ..errors import InvalidDataDefinitionException
import shutil
import pathlib
from dask_ml.linear_model import LinearRegression
from divina.divina.models.utils import compare_sk_models
import joblib
import pandas as pd
import dask.dataframe as ddf
from ..aws.utils import create_modelling_emr


def test_validate_vision_definition(
    vd_no_target,
    vd_time_horizons_not_list,
    vd_time_validation_splits_not_list,
    vd_no_time_index,
    vd_no_dataset_id,
    vd_no_dataset_directory,
):
    for dd in [
        vd_no_target,
        vd_no_time_index,
        vd_time_validation_splits_not_list,
        vd_time_horizons_not_list,
        vd_no_dataset_id,
        vd_no_dataset_directory,
    ]:
        try:
            validate_vision_definition(dd)
        except InvalidDataDefinitionException:
            assert True
        else:
            assert False


@patch("s3fs.S3FileSystem.open", open)
@patch("s3fs.S3FileSystem.ls", os.listdir)
@patch.dict(os.environ, {"DATA_BUCKET": "divina-test/data"})
@patch.dict(os.environ, {"DATASET_BUCKET": "divina-test/dataset"})
@patch.dict(os.environ, {"DATASET_ID": "test1"})
def test_dataset_build(s3_fs, vision_s3, test_df_1, account_number):
    pathlib.Path(os.environ["DATA_BUCKET"]).mkdir(parents=True, exist_ok=True)
    test_df_1.to_csv(
        os.path.join(os.environ["DATA_BUCKET"], "test_df_1.csv"), index=False
    )
    pathlib.Path(os.environ["DATASET_BUCKET"], os.environ["DATASET_ID"]).mkdir(
        parents=True, exist_ok=True
    )
    build_dataset_dask(
        s3_fs=s3_fs,
        read_path=os.environ["DATA_BUCKET"],
        write_path=os.environ["DATASET_BUCKET"],
        dataset_name=os.environ["DATASET_ID"],
        partition_dimensions=None,
    )

    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                os.environ["DATASET_BUCKET"], os.environ["DATASET_ID"], "data/*"
            )
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


@patch("s3fs.S3FileSystem.open", open)
@patch("s3fs.S3FileSystem.ls", os.listdir)
@patch.dict(os.environ, {"VISION_BUCKET": "divina-test/vision"})
@patch.dict(os.environ, {"VISION_ID": "test1"})
def test_dask_train(s3_fs, test_df_1, test_vd_1, test_model_1, dask_client):
    pathlib.Path(
        os.path.join(
            test_vd_1["vision_definition"]["dataset_directory"],
            "{}".format(test_vd_1["vision_definition"]["dataset_id"]),
            "data",
        )
    ).mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        os.path.join(
            test_vd_1["vision_definition"]["dataset_directory"],
            "{}".format(test_vd_1["vision_definition"]["dataset_id"]),
            "profile",
        )
    ).mkdir(parents=True, exist_ok=True)
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
    dask_train(
        s3_fs=s3_fs,
        dask_model=LinearRegression,
        vision_definition=test_vd_1["vision_definition"],
        vision_id=os.environ["VISION_ID"],
        write_path=os.environ["VISION_BUCKET"],
    )

    assert compare_sk_models(
        joblib.load(
            os.path.abspath(
                os.path.join(
                    os.environ["VISION_BUCKET"],
                    os.environ["VISION_ID"],
                    "models",
                    "s-19700101-000008_h-1",
                )
            )
        ),
        test_model_1,
    )


@patch("s3fs.S3FileSystem.open", open)
@patch("s3fs.S3FileSystem.ls", os.listdir)
def test_get_composite_dataset(
    test_df_1,
    test_df_2,
    test_vd_2,
    test_composite_profile_1,
    test_composite_dataset_1,
    dask_client,
):
    for path in ["data", "profile"]:
        for dataset in test_vd_2["vision_definition"]["joins"]:
            pathlib.Path(
                os.path.join(
                    dataset["dataset_directory"],
                    "{}".format(dataset["dataset_id"]),
                    path,
                )
            ).mkdir(parents=True, exist_ok=True)
        pathlib.Path(
            os.path.join(
                test_vd_2["vision_definition"]["dataset_directory"],
                "{}".format(test_vd_2["vision_definition"]["dataset_id"]),
                path,
            )
        ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_1, chunksize=10000).to_parquet(
        os.path.join(
            test_vd_2["vision_definition"]["dataset_directory"],
            test_vd_2["vision_definition"]["dataset_id"],
            "data",
        )
    )
    ddf.from_pandas(test_df_1.describe(), chunksize=10000).to_parquet(
        os.path.join(
            test_vd_2["vision_definition"]["dataset_directory"],
            test_vd_2["vision_definition"]["dataset_id"],
            "profile",
        )
    )
    ddf.from_pandas(test_df_2, chunksize=10000).to_parquet(
        os.path.join(
            test_vd_2["vision_definition"]["joins"][0]["dataset_directory"],
            test_vd_2["vision_definition"]["joins"][0]["dataset_id"],
            "data",
        )
    )
    ddf.from_pandas(test_df_2.describe(), chunksize=10000).to_parquet(
        os.path.join(
            test_vd_2["vision_definition"]["joins"][0]["dataset_directory"],
            test_vd_2["vision_definition"]["joins"][0]["dataset_id"],
            "profile",
        )
    )
    df, profile = get_dataset(test_vd_2["vision_definition"])

    pd.testing.assert_frame_equal(df.compute(), test_composite_dataset_1)
    pd.testing.assert_frame_equal(profile.compute(), test_composite_profile_1)


@patch("s3fs.S3FileSystem.open", open)
@patch("s3fs.S3FileSystem.ls", os.listdir)
@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(os.environ, {"VISION_BUCKET": "divina-test/vision"})
def test_dask_predict(
    s3_fs, dask_client, test_df_1, test_vd_1, test_model_1, test_predictions_1
):
    pathlib.Path(
        os.path.join(
            test_vd_1["vision_definition"]["dataset_directory"],
            test_vd_1["vision_definition"]["dataset_id"],
            "data",
        )
    ).mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        os.path.join(os.environ["VISION_BUCKET"], os.environ["VISION_ID"], "models")
    ).mkdir(parents=True, exist_ok=True)
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
    joblib.dump(
        test_model_1,
        os.path.join(
            os.environ["VISION_BUCKET"],
            os.environ["VISION_ID"],
            "models",
            "s-19700101-000008_h-1",
        ),
    )
    with open(
        os.path.join(
            os.environ["VISION_BUCKET"],
            os.environ["VISION_ID"],
            "vision_definition.json",
        ),
        "w+",
    ) as f:
        json.dump(test_vd_1, f)

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


@patch("s3fs.S3FileSystem.open", open)
@patch("s3fs.S3FileSystem.ls", os.listdir)
@patch.dict(os.environ, {"VISION_ID": "test1"})
@patch.dict(os.environ, {"VISION_BUCKET": "divina-test/vision"})
def test_dask_validate(
    s3_fs, test_vd_1, test_df_1, test_metrics_1, test_predictions_1, dask_client
):
    ddf.from_pandas(test_predictions_1, chunksize=10000).to_parquet(
        os.path.join(
            os.environ["VISION_BUCKET"],
            os.environ["VISION_ID"],
            "predictions",
            "s-19700101-000008",
        )
    )
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

    dask_validate(
        s3_fs=s3_fs,
        vision_definition=test_vd_1["vision_definition"],
        vision_id=os.environ["VISION_ID"],
        read_path=os.environ["VISION_BUCKET"],
        write_path=os.environ["VISION_BUCKET"],
    )

    with open(
        os.path.join(
            os.environ["VISION_BUCKET"], os.environ["VISION_ID"], "metrics.json"
        ),
        "r",
    ) as f:
        metrics = json.load(f)

    assert metrics == test_metrics_1
