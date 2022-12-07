import json
import os
import pathlib

import dask.dataframe as ddf
import joblib
import pandas as pd
import numpy as np

import s3fs
import plotly.graph_objects as go

from pipeline.pipeline import Pipeline, PipelineValidation, ValidationSplit, BoostValidation, CausalValidation
from pipeline.utils import get_parameters, set_parameters


def test_bin_features(
        test_data_1,
        test_df_1,
        test_pipeline_1,
):
    df = test_pipeline_1.preprocess(test_data_1)
    pd.testing.assert_frame_equal(
        df.compute(),
        test_df_1.compute().set_index('a'),
    )


###TODO - update fixtures to match this output
def test_glm_train(
        test_df_1,
        test_pipeline_1,
        test_model_1,
        random_state,
):
    model = test_pipeline_1.train(
        x=test_df_1.set_index('a').drop(columns='c'), y=test_df_1[['c']], random_state=random_state, model_type='GLM',
        model_params={'link_function': 'log'})
    assert model == test_model_1[0]


def test_glm_forecast(
        test_df_1,
        test_pipeline_1,
        test_model_1,
        test_forecast_1,
):
    result = test_pipeline_1.forecast(
        model=test_model_1[0], x=test_df_1.set_index('a').drop(columns='c')
    )
    ##TODO - start here figure out why forecasts and test values are so bad and then resume changing components to new format
    np.testing.assert_equal(
        result.compute().values,
        test_forecast_1.compute().values
    )


def test_validate(
        test_pipeline_1,
        test_df_1,
        test_forecast_1,
        test_metrics_1
):
    metrics = test_pipeline_1.validate(truth_dataset=test_df_1[['c']], prediction_dataset=test_forecast_1)
    assert metrics == test_metrics_1


def test_get_params(test_model_1, test_params_1):
    pipeline_path = "divina-test/pipeline/test1"
    pathlib.Path(os.path.join(pipeline_path, "models")).mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(
        test_model_1,
        os.path.join(
            pipeline_path,
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    with open(
            os.path.join(
                pipeline_path,
                "models",
                "s-19700101-000007_h-1_params",
            ),
            "w+",
    ) as f:
        json.dump(test_model_1[1], f)
    params = get_parameters(
        model_path=os.path.join(
            pipeline_path,
            "models",
            "s-19700101-000007_h-1",
        )
    )

    assert params == test_params_1


def test_set_params(test_model_1, test_params_1, test_params_2):
    pipeline_path = "divina-test/pipeline/test1"
    pathlib.Path(os.path.join(pipeline_path, "models")).mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(
        test_model_1,
        os.path.join(
            pipeline_path,
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    with open(
            os.path.join(
                pipeline_path,
                "models",
                "s-19700101-000007_h-1_params",
            ),
            "w+",
    ) as f:
        json.dump(test_params_1, f)
    set_parameters(
        model_path=os.path.join(
            pipeline_path,
            "models",
            "s-19700101-000007_h-1",
        ),
        params=test_params_2["features"],
    )

    with open(
            os.path.join(
                pipeline_path,
                "models",
                "s-19700101-000007_h-1_params",
            ),
            "rb",
    ) as f:
        params = json.load(f)

    assert params == test_params_2


def test_pipeline(
        test_df_4,
        test_pipeline_2,
        test_pipeline_result
):
    result = test_pipeline_2.fit(test_df_4)
    assert result == test_pipeline_result


def test_simulation_pipeline_prefect(
        test_df_4,
        test_pipeline_2,
        test_pipeline_result,
        test_boost_model_params,
        test_bucket,
        test_pipeline_root,
        test_pipeline_name,
        test_bootstrap_models,
        test_boost_models,
        test_horizons

):
    test_pipeline_2.env_variables = {'AWS_SECRET_ACCESS_KEY': os.environ['AWS_SECRET_ACCESS_KEY'],
                       'AWS_ACCESS_KEY_ID': os.environ['AWS_ACCESS_KEY_ID']}
    test_pipeline_2.storage_options = {'client_kwargs': {'endpoint_url': 'http://127.0.0.1:{}'.format(9000)}}
    test_pipeline_2.is_fit = True
    test_pipeline_2.bootstrap_models = test_bootstrap_models
    test_pipeline_2.boost_models = test_boost_models
    test_data_path = '{}/test-data'.format(test_pipeline_root)
    fs = s3fs.S3FileSystem(**test_pipeline_2.storage_options)
    if fs.exists(test_bucket):
        fs.rm(test_bucket, True)
    else:
        fs.mkdir(test_bucket)
    test_df_4.to_parquet(test_data_path,
                         storage_options={'client_kwargs': {'endpoint_url': 'http://127.0.0.1:{}'.format(9000)}})
    from prefect import flow
    @flow(
        name=test_pipeline_name, persist_result=True
    )
    def run_pipeline(df: str):
        return test_pipeline_2.simulate(df=df, scenarios={
                "b": {"mode": "constant", "constant_values": [0, 1, 2, 3, 4, 5]}
            }, horizons=test_horizons, prefect=True)

    result = run_pipeline(test_data_path)
    assert result == test_pipeline_result

'''def test_example_pipeline_kfp(
        test_df_4,
        test_ed_2,
        test_pipeline_result,
        test_kind_cluster,
        test_boost_model_params,
        test_bucket,
        minio_endpoint,
        test_pipeline_root,
        test_pipeline_name,

):
    pipeline = Pipeline(**test_ed_2["pipeline_definition"])

    with test_kind_cluster.port_forward("service/ml-pipeline-ui", 80, "-n", "kubeflow", retries=1) as port:
        with test_kind_cluster.port_forward("service/minio-service", 9000, "-n", "kubeflow", retries=1) as minio_port:
            @pipeline(
                name=test_pipeline_name,
                description="testing",
                pipeline_root=test_pipeline_root,
            )
            def run_pipeline(data: str = 'test_uri', boost_model_params=json.dumps({"test": "test2"})):
                storage_options['client_kwargs'].update({'endpoint_url': minio_endpoint})
                component_kwargs = {'storage_options': storage_options}
                pipeline.train(data=data, boost_model_params=boost_model_params,
                               env_variables={'AWS_ACCESS_KEY_ID': 'minio', 'AWS_SECRET_ACCESS_KEY': 'minio123'},
                               component_kwargs=component_kwargs,
                               kfp=True)

            import sys
            import s3fs
            sys.stderr.write('Kubeflow listening on 127.0.0.1:{}\n'.format(port))
            sys.stderr.write('Minio listening on 127.0.0.1:{}\n'.format(minio_port))
            storage_options = {'client_kwargs': {'endpoint_url': 'http://127.0.0.1:{}'.format(minio_port)}}
            test_data_path = '{}/test-data'.format(test_pipeline_root)
            fs = s3fs.S3FileSystem(**storage_options)
            if fs.exists(test_bucket):
                fs.rm(test_bucket, True)
            else:
                fs.mkdir(test_bucket)
            test_df_4.to_parquet(test_data_path, storage_options=storage_options)
            client = kfp.Client(host='http://127.0.0.1:{}'.format(port))
            client.create_run_from_pipeline_func(run_pipeline,
                                                 arguments={'data': test_data_path,
                                                            "boost_model_params": json.dumps(
                                                                test_boost_model_params)})

            pass
            # with fs.open(pipeline_root + '/result.pkl', 'rb') as f:
            #    result = dill.load(f)
            # assert result == test_pipeline_result

        ###TODO start here - build result object from pipeline-root and compare - create parse method in result object
        ###TODO start here - test kfp pipeline with kind
        ###TODO then implement interpretability/analytics interface'''


def test_pipeline_prefect(
        test_df_4,
        test_pipeline_2,
        test_pipeline_result,
        test_boost_model_params,
        test_bucket,
        test_pipeline_root,
        test_pipeline_name,

):
    test_pipeline_2.env_variables = {'AWS_SECRET_ACCESS_KEY': os.environ['AWS_SECRET_ACCESS_KEY'],
                       'AWS_ACCESS_KEY_ID': os.environ['AWS_ACCESS_KEY_ID']}
    test_pipeline_2.storage_options = {'client_kwargs': {'endpoint_url': 'http://127.0.0.1:{}'.format(9000)}}
    test_data_path = '{}/test-data'.format(test_pipeline_root)
    fs = s3fs.S3FileSystem(**test_pipeline_2.storage_options)
    if fs.exists(test_bucket):
        fs.rm(test_bucket, True)
    else:
        fs.mkdir(test_bucket)
    test_df_4.to_parquet(test_data_path,
                         storage_options={'client_kwargs': {'endpoint_url': 'http://127.0.0.1:{}'.format(9000)}})
    from prefect import flow
    @flow(
        name=test_pipeline_name, persist_result=True
    )
    def run_pipeline(df: str):
        return test_pipeline_2.fit(df=df, prefect=True)

    result = run_pipeline(test_data_path)
    assert result == test_pipeline_result

###TODO start here - reset hardcoded values in conftest for this test - use same fixtures for unit and pieline tests
###TODO then implement interpretability/analytics interface

def test_quickstart(test_eds_quickstart, random_state):
    for k in test_eds_quickstart:
        ed = test_eds_quickstart[k]
        pipeline_path = "divina-test/pipeline/test1"
        pipeline = Pipeline(**ed["pipeline_definition"])
        result = pipeline.train(write_path=pipeline_path, random_state=11)
        result_df = result.compute().reset_index(drop=True)
        ###RESET
        """ddf.read_parquet(
            os.path.join(
                pipeline_path,
                "forecast"
            )
        ).to_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'docs_src/results/forecasts',
                               k))"""
        pd.testing.assert_frame_equal(
            result_df,
            ddf.read_parquet(
                pathlib.Path(
                    pathlib.Path(__file__).parent.parent.parent,
                    "docs_src/results/forecasts",
                    k,
                )
            )
            .compute()
            .reset_index(drop=True),
        )
        ed["pipeline_definition"]["time_horizons"] = [0]
        if not "target_dimensions" in ed["pipeline_definition"]:
            stores = [6]
        else:
            stores = [1, 2, 3]
        result_df = result_df[result_df["Date"] >= "2015-01-01"]
        for s in stores:
            fig = go.Figure()
            for h in ed["pipeline_definition"]["time_horizons"]:
                if not "encode_features" in ed["pipeline_definition"]:
                    store_df = result_df[result_df["Store"] == s]
                else:
                    store_df = result_df[result_df["Store_{}".format(float(s))] == 1]
                if "scenarios" in ed["pipeline_definition"]:
                    store_df = store_df[
                        (store_df["Date"] < "2015-08-01") | (result_df["Promo"] == 1)
                        ]
                    fig.add_vrect(
                        x0=pd.to_datetime("07-31-2015").timestamp() * 1000,
                        x1=pd.to_datetime("01-01-2016").timestamp() * 1000,
                        line_width=2,
                        line_color="cadetblue",
                        annotation_text="Forecasts assuming promotions",
                    )
                if "confidence_intervals" in ed["pipeline_definition"]:
                    if len(ed["pipeline_definition"]["confidence_intervals"]) > 0:
                        for i in ed["pipeline_definition"]["confidence_intervals"]:
                            fig.add_trace(
                                go.Scatter(
                                    marker=dict(color="cyan"),
                                    mode="lines",
                                    x=store_df[
                                        ed["pipeline_definition"]["time_index"]
                                    ],
                                    y=store_df[
                                        "{}_h_{}_pred_c_{}".format(
                                            ed["pipeline_definition"]["target"], h, i
                                        )
                                    ],
                                    name="h_{}_c_{}".format(h, i),
                                )
                            )
                        fig.update_traces(
                            fill="tonexty",
                            name="Confidence Bound",
                            selector=dict(
                                name="h_{}_c_{}".format(
                                    h,
                                    ed["pipeline_definition"]["confidence_intervals"][
                                        -1
                                    ],
                                )
                            ),
                        )
                        fig.update_traces(
                            showlegend=False,
                            name="Upper Confidence Bound",
                            selector=dict(
                                name="h_{}_c_{}".format(
                                    h,
                                    ed["pipeline_definition"]["confidence_intervals"][
                                        0
                                    ],
                                )
                            ),
                        )
                fig.add_trace(
                    go.Scatter(
                        marker=dict(color="black"),
                        line=dict(dash="dash"),
                        mode="lines",
                        x=store_df[ed["pipeline_definition"]["time_index"]],
                        y=store_df[ed["pipeline_definition"]["target"]],
                        name=ed["pipeline_definition"]["target"],
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        marker=dict(color="darkblue"),
                        mode="lines",
                        x=store_df[ed["pipeline_definition"]["time_index"]],
                        y=store_df[
                            "{}_h_{}_pred".format(
                                ed["pipeline_definition"]["target"], h
                            )
                        ],
                        name="Forecast".format(h),
                    )
                )
                path = pathlib.Path(
                    pathlib.Path(__file__).parent.parent.parent,
                    "docs_src/_static/plots/quickstart/{}_h_{}_s_{}_2d.html".format(
                        k, h, s
                    ),
                )
                fig.update_layout(legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.02,
                    xanchor="center",
                    x=0.5
                ))
                fig.update_xaxes(side="top")
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(path)
                factor_fig = go.Figure()
                factors = [c for c in store_df if c.split("_")[0] == "factor"]
                if "scenarios" in ed["pipeline_definition"]:
                    store_df = store_df[
                        (store_df["Date"] > "2015-08-01") | (result_df["Promo"] == 1)
                        ]
                store_df = store_df[
                    factors + [ed["pipeline_definition"]["time_index"]]
                    ]
                for f in factors:
                    factor_fig.add_trace(
                        go.Bar(
                            x=store_df[ed["pipeline_definition"]["time_index"]],
                            y=store_df[f],
                            name=("_".join(f.split("_")[1:])[:15] + "..")
                            if len("_".join(f.split("_")[1:])) > 17
                            else "_".join(f.split("_")[1:]),
                        )
                    )
                factor_fig.update_layout(barmode="relative", legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.05,
                    xanchor="center",
                    x=0.5,
                ))
                factor_fig.update_xaxes(side="top")
                path = pathlib.Path(
                    pathlib.Path(__file__).parent.parent.parent,
                    "docs_src/_static/plots/quickstart/{}_test_forecast_retail_h_{}_s_{}_factors.html".format(
                        k, h, s
                    ),
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                factor_fig.write_html(path)
