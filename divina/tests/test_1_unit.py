import os
import json
from ..train import _train
from ..forecast import _forecast
from ..dataset import _get_dataset
from ..validate import _validate
from ..experiment import _experiment
import pathlib
from dask_ml.linear_model import LinearRegression
from ..utils import compare_sk_models, get_parameters, set_parameters
import joblib
import pandas as pd
import dask.dataframe as ddf
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import plotly.graph_objects as go


def test_validate_experiment_definition(
        fd_no_target,
        fd_time_horizons_not_list,
        fd_time_validation_splits_not_list,
        fd_no_time_index,
        fd_no_data_path,
        fd_invalid_model,
        fd_time_horizons_range_not_tuple
):
    for dd in [
        fd_no_target,
        fd_no_time_index,
        fd_time_validation_splits_not_list,
        fd_time_horizons_not_list,
        fd_no_data_path,
        fd_invalid_model,
        fd_time_horizons_range_not_tuple
    ]:
        try:
            with open(pathlib.Path(pathlib.Path(__file__).parent.parent, 'config/fd_schema.json'), 'r') as f:
                validate(instance=dd, schema=json.load(f))
        except ValidationError:
            return None
        else:
            assert False


def test_get_composite_dataset(
        test_df_1,
        test_df_2,
        test_fd_2,
        test_composite_dataset_1,
        dask_client,
):
    for dataset in test_fd_2["experiment_definition"]["joins"]:
        pathlib.Path(
            dataset["data_path"]
        ).mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        test_fd_2["experiment_definition"]["data_path"],
    ).mkdir(parents=True, exist_ok=True)

    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_2["experiment_definition"]["data_path"]
    )
    ddf.from_pandas(test_df_2, npartitions=2).to_parquet(

        test_fd_2["experiment_definition"]["joins"][0]["data_path"]
    )
    df = _get_dataset(test_fd_2["experiment_definition"])

    pd.testing.assert_frame_equal(df.compute().reset_index(drop=True), test_composite_dataset_1.reset_index(drop=True))


def test_train(s3_fs, test_df_1, test_fd_1, test_model_1, dask_client, test_bootstrap_models, test_validation_models,
               random_state):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        test_fd_1["experiment_definition"]["data_path"],
    ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_1["experiment_definition"]["data_path"]
    )
    _train(
        s3_fs=s3_fs,
        dask_model=LinearRegression,
        experiment_definition=test_fd_1["experiment_definition"],
        write_path=vision_path,
        random_seed=random_state,
    )

    compare_sk_models(
        joblib.load(
            os.path.abspath(
                os.path.join(
                    vision_path,
                    "models",
                    "h-1",
                )
            )
        ),
        test_model_1[0],
    )
    with open(os.path.abspath(
            os.path.join(
                vision_path,
                "models",
                "h-1_params.json",
            )
    )) as f:
        assert json.load(f) == test_model_1[1]
    for seed in test_bootstrap_models:
        compare_sk_models(
            joblib.load(
                os.path.abspath(
                    os.path.join(
                        vision_path,
                        "models/bootstrap",
                        "h-1_r-{}".format(seed),
                    )
                )
            ),
            test_bootstrap_models[seed][0],
        )
        with open(os.path.abspath(
                os.path.join(
                    vision_path,
                    "models/bootstrap",
                    "h-1_r-{}_params.json".format(seed),
                )
        )) as f:
            assert json.load(f) == test_bootstrap_models[seed][1]
    for split in test_validation_models:
        compare_sk_models(
            joblib.load(
                os.path.abspath(
                    os.path.join(
                        vision_path,
                        "models",
                        "s-{}_h-1".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
                    )
                )
            ),
            test_validation_models[split][0],
        )
        with open(os.path.abspath(
                os.path.join(
                    vision_path,
                    "models",
                    "s-{}_h-1_params.json".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
                )
        )) as f:
            features = json.load(f)
        assert features == test_validation_models[split][1]


def test_train_retail(s3_fs, test_df_retail_sales, test_df_retail_stores, test_df_retail_time, test_fd_retail,
                      test_model_retail, dask_client, test_validation_models_retail, test_bootstrap_models_retail,
                      random_state):
    vision_path = "divina-test/vision/test1"
    _train(
        s3_fs=s3_fs,
        dask_model=LinearRegression,
        experiment_definition=test_fd_retail["experiment_definition"],
        write_path=vision_path,
        random_seed=random_state,
    )
    compare_sk_models(
        joblib.load(
            os.path.abspath(
                os.path.join(
                    vision_path,
                    "models",
                    "h-2",
                )
            )
        ),
        test_model_retail[0],
    )
    with open(os.path.abspath(
            os.path.join(
                vision_path,
                "models",
                "h-2_params.json",
            )
    )) as f:
        assert json.load(f) == test_model_retail[1]
    for seed in test_bootstrap_models_retail:
        compare_sk_models(
            joblib.load(
                os.path.abspath(
                    os.path.join(
                        vision_path,
                        "models/bootstrap",
                        "h-2_r-{}".format(seed),
                    )
                )
            ),
            test_bootstrap_models_retail[seed][0],
        )
        with open(os.path.abspath(
                os.path.join(
                    vision_path,
                    "models/bootstrap",
                    "h-2_r-{}_params.json".format(seed),
                )
        )) as f:
            features = json.load(f)
        assert features == test_bootstrap_models_retail[seed][1]
    for split in test_validation_models_retail:
        compare_sk_models(
            joblib.load(
                os.path.abspath(
                    os.path.join(
                        vision_path,
                        "models",
                        "s-{}_h-2".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
                    )
                )
            ),
            test_validation_models_retail[split][0],
        )
        with open(os.path.abspath(
                os.path.join(
                    vision_path,
                    "models",
                    "s-{}_h-2_params.json".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
                )
        )) as f:
            features = json.load(f)
        assert features == test_validation_models_retail[split][1]


def test_forecast(
        s3_fs, dask_client, test_df_1, test_fd_1, test_model_1, test_val_predictions_1, test_forecast_1,
        test_bootstrap_models
):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        test_fd_1["experiment_definition"]["data_path"]
    ).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(vision_path, "models/bootstrap")).mkdir(
        parents=True, exist_ok=True
    )
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_1["experiment_definition"]["data_path"]
    )
    joblib.dump(
        test_model_1[0],
        os.path.join(
            vision_path,
            "models",
            "h-1",
        ),
    )
    with open(os.path.join(
            vision_path,
            "models",
            "h-1_params.json",
    ), 'w+') as f:
        json.dump(
            test_model_1[1],
            f
        )
    for seed in test_bootstrap_models:
        joblib.dump(
            test_bootstrap_models[seed][0],
            os.path.join(
                vision_path,
                "models/bootstrap",
                "h-1_r-{}".format(seed),
            ),
        )
        with open(os.path.join(
                vision_path,
                "models/bootstrap",
                "h-1_r-{}_params.json".format(seed),
        ), 'w+') as f:
            json.dump(
                test_bootstrap_models[seed][1],
                f
            )

    _forecast(
        s3_fs=s3_fs,
        experiment_definition=test_fd_1["experiment_definition"],
        read_path=vision_path,
        write_path=vision_path,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "forecast"
            )
        ).compute().reset_index(drop=True),
        test_forecast_1.reset_index(drop=True), check_dtype=False
    )


def test_forecast_retail(s3_fs, test_df_retail_sales, test_df_retail_stores, test_df_retail_time, test_fd_retail,
                         test_model_retail, test_val_predictions_retail, test_forecast_retail,
                         test_bootstrap_models_retail, dask_client, random_state):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        os.path.join(
            vision_path, "models", "bootstrap"
        )
    ).mkdir(parents=True, exist_ok=True)
    joblib.dump(
        test_model_retail[0],
        os.path.join(
            vision_path,
            "models",
            "h-2",
        ),
    )
    with open(os.path.join(
            vision_path,
            "models",
            "h-2_params.json",
    ), 'w+') as f:
        json.dump(
            test_model_retail[1],
            f
        )
    for seed in test_bootstrap_models_retail:
        joblib.dump(
            test_bootstrap_models_retail[seed][0],
            os.path.join(
                vision_path,
                "models/bootstrap",
                "h-2_r-{}".format(seed),
            ),
        )
        with open(os.path.join(
                vision_path,
                "models/bootstrap",
                "h-2_r-{}_params.json".format(seed),
        ), 'w+') as f:
            json.dump(
                test_bootstrap_models_retail[seed][1],
                f
            )
    _forecast(
        s3_fs=s3_fs,
        experiment_definition=test_fd_retail["experiment_definition"],
        read_path=vision_path,
        write_path=vision_path
    )
    result_df = ddf.read_parquet(
        os.path.join(
            vision_path,
            "forecast"
        )
    ).compute().reset_index(drop=True)
    pathlib.Path(
        "docs_src/plots"
    ).mkdir(parents=True, exist_ok=True)
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text="Insample and Blind Forecasts")
        )
    )
    for h in test_fd_retail["experiment_definition"]['time_horizons']:
        result_df_2d = result_df.sort_values('Promo').groupby('Date').agg('first').reset_index()
        for i in test_fd_retail["experiment_definition"]['confidence_intervals']:
            fig.add_trace(go.Scatter(marker=dict(color="cyan"), mode="lines",
                                     x=result_df_2d[test_fd_retail["experiment_definition"]['time_index']],
                                     y=result_df_2d[
                                         '{}_h_{}_pred_c_{}'.format(test_fd_retail["experiment_definition"]['target'],
                                                                    h, i)], name="h_{}_c_{}".format(h, i)))
        fig.update_traces(fill='tonexty', name='Confidence Bound', selector=dict(
            name="h_{}_c_{}".format(h, test_fd_retail["experiment_definition"]['confidence_intervals'][-1])))
        fig.update_traces(showlegend=False, name='Upper Confidence Bound', selector=dict(
            name="h_{}_c_{}".format(h, test_fd_retail["experiment_definition"]['confidence_intervals'][0])))
        fig.add_trace(
            go.Scatter(marker=dict(color='black'), line=dict(dash='dash'), mode="lines",
                       x=result_df_2d[test_fd_retail["experiment_definition"]['time_index']],
                       y=result_df_2d[test_fd_retail["experiment_definition"]['target']],
                       name=test_fd_retail["experiment_definition"]['target']))
        fig.add_trace(go.Scatter(marker=dict(color="darkblue"), mode="lines",
                                 x=result_df_2d[test_fd_retail["experiment_definition"]['time_index']],
                                 y=result_df_2d[
                                     '{}_h_{}_pred'.format(test_fd_retail["experiment_definition"]['target'],
                                                           h)], name="Horizon {} Forecast".format(h)))
        fig.add_vrect(x0=pd.to_datetime('07-31-2015').timestamp() * 1000,
                      x1=pd.to_datetime('08-30-2015').timestamp() * 1000, line_width=2, line_color="cadetblue",
                      annotation_text='Blind Forecasts with Promotions Simulated as False')
        fig.write_html('docs_src/plots/test_forecast_retail_h_{}_2d.html'.format(h))
    fig = go.Figure()
    for h in test_fd_retail["experiment_definition"]['time_horizons']:
        result_df_2d = result_df[
            (result_df['Promo'] == 1) & (result_df['Store_1.0'] == 1) & (result_df['Date'] >= '2015-08-01')]
        factors = [c for c in result_df_2d if c.split("_")[0] == "factor"]
        result_df_2d = result_df_2d[factors + [test_fd_retail["experiment_definition"]['time_index']]]
        for f in factors:
            fig.add_trace(go.Bar(x=result_df_2d[test_fd_retail["experiment_definition"]['time_index']],
                                 y=result_df_2d[
                                     f], name="_".join(f.split("_")[1:])))
            fig.update_layout(barmode='relative', title_text='Factor')
        fig.write_html('docs_src/plots/test_forecast_retail_h_{}_factors.html'.format(h))
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text="Forecasted Promo Impact"),
            scene=dict(
                xaxis=dict(
                    title='Date'),
                yaxis=dict(
                    title='Promo', tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes']),
                zaxis=dict(
                    title='Sales'))
        )
    )
    for h in test_fd_retail["experiment_definition"]['time_horizons']:
        result_df_3d = result_df[(result_df[test_fd_retail["experiment_definition"]['time_index']] >= '08-01-2015') & (
                    result_df['Store_1.0'] == 1)].sort_values('Date')
        fig.add_trace(go.Surface(x=result_df_3d[test_fd_retail["experiment_definition"]['time_index']].unique(),
                                 z=[result_df_3d[result_df_3d['Promo'] == 0][
                                        '{}_h_{}_pred'.format(test_fd_retail["experiment_definition"]['target'],
                                                              h)].values, result_df_3d[result_df_3d['Promo'] == 1][
                                        '{}_h_{}_pred'.format(test_fd_retail["experiment_definition"]['target'],
                                                              h)].values],
                                 y=result_df_3d['Promo'].unique(), opacity=.4))
        fig.write_html('docs_src/plots/test_forecast_retail_h_{}_3d.html'.format(h))
    pd.testing.assert_frame_equal(
        result_df,
        test_forecast_retail.reset_index(drop=True), check_dtype=False
    )


def test_validate(
        s3_fs, test_fd_1, test_df_1, test_metrics_1, dask_client, test_val_predictions_1, test_validation_models,
        test_model_1,
        test_bootstrap_models
):
    vision_path = "divina-test/vision/test1"
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_fd_1["experiment_definition"]["data_path"]
    )
    pathlib.Path(
        os.path.join(
            vision_path, "models", "bootstrap"
        )
    ).mkdir(parents=True, exist_ok=True)

    for split in test_validation_models:
        joblib.dump(
            test_validation_models[split][0],
            os.path.join(
                vision_path,
                "models",
                "s-{}_h-1".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
            ),
        )
        with open(os.path.join(
                vision_path,
                "models",
                "s-{}_h-1_params.json".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
        ), 'w+') as f:
            json.dump(
                test_validation_models[split][1],
                f
            )
    for seed in test_bootstrap_models:
        joblib.dump(
            test_bootstrap_models[seed][0],
            os.path.join(
                vision_path,
                "models/bootstrap",
                "h-1_r-{}".format(seed),
            ),
        )
        with open(os.path.join(
                vision_path,
                "models/bootstrap",
                "h-1_r-{}_params.json".format(seed),
        ), 'w+') as f:
            json.dump(
                test_bootstrap_models[seed][1],
                f
            )
    _validate(
        s3_fs=s3_fs,
        experiment_definition=test_fd_1["experiment_definition"],
        read_path=vision_path,
        write_path=vision_path,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "validation",
                "s-19700101-000007",
            )
        ).compute().reset_index(drop=True),
        test_val_predictions_1.reset_index(drop=True), check_dtype=False
    )
    with open(
            os.path.join(vision_path, "metrics.json"),
            "r",
    ) as f:
        assert json.load(f) == test_metrics_1


def test_validate_retail(s3_fs, test_df_retail_sales, test_df_retail_stores, test_df_retail_time, test_fd_retail,
                         test_val_predictions_retail, test_metrics_retail, test_model_retail,
                         test_bootstrap_models_retail, test_validation_models_retail, dask_client):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(
        os.path.join(
            vision_path, "models", "bootstrap"
        )
    ).mkdir(parents=True, exist_ok=True)
    for split in test_fd_retail["experiment_definition"]["time_validation_splits"]:
        joblib.dump(
            test_validation_models_retail[split][0],
            os.path.join(
                vision_path,
                "models",
                "s-{}_h-2".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
            ),
        )
        with open(os.path.join(
                vision_path,
                "models",
                "s-{}_h-2_params.json".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
        ), 'w+') as f:
            json.dump(
                test_validation_models_retail[split][1],
                f
            )
    for seed in test_bootstrap_models_retail:
        joblib.dump(
            test_bootstrap_models_retail[seed][0],
            os.path.join(
                vision_path,
                "models/bootstrap",
                "s-20150718-000000_h-2_r-{}".format(seed),
            ),
        )
        with open(os.path.join(
                vision_path,
                "models/bootstrap",
                "s-20150718-000000_h-2_r-{}_params.json".format(seed),
        ), 'w+') as f:
            json.dump(
                test_bootstrap_models_retail[seed][1],
                f
            )
    _validate(
        s3_fs=s3_fs,
        experiment_definition=test_fd_retail["experiment_definition"],
        read_path=vision_path,
        write_path=vision_path
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                vision_path,
                "validation",
                "s-20150718-000000",
            )
        ).compute().reset_index(drop=True),
        test_val_predictions_retail.reset_index(drop=True), check_dtype=False
    )
    with open(
            os.path.join(vision_path, "metrics.json"),
            "r",
    ) as f:
        metrics = json.load(f)
    assert metrics == test_metrics_retail


def test_get_params(
        s3_fs, test_model_1, test_params_1
):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(os.path.join(vision_path, "models")).mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(
        test_model_1,
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    with open(os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1_params",
    ), 'w+') as f:
        json.dump(test_model_1[1], f)
    params = get_parameters(s3_fs=s3_fs, model_path=os.path.join(
        vision_path,
        "models",
        "s-19700101-000007_h-1",
    ))

    assert params == test_params_1


def test_set_params(
        s3_fs, test_model_1, test_params_1, test_params_2
):
    vision_path = "divina-test/vision/test1"
    pathlib.Path(os.path.join(vision_path, "models")).mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(
        test_model_1,
        os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    with open(os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1_params",
    ), 'w+') as f:
        json.dump(test_params_1, f)
    set_parameters(s3_fs=s3_fs, model_path=os.path.join(
        vision_path,
        "models",
        "s-19700101-000007_h-1",
    ), params=test_params_2['features'])

    with open(os.path.join(
            vision_path,
            "models",
            "s-19700101-000007_h-1_params",
    ), 'rb') as f:
        params = json.load(f)

    assert params == test_params_2


def test_experiment_retail_min(test_fd_retail_min, random_state):
    vision_path = "divina-test/vision/test1"
    _experiment(
        experiment_definition=test_fd_retail_min["experiment_definition"],
        read_path=vision_path,
        write_path=vision_path
    )
    result_df = ddf.read_parquet(
        os.path.join(
            vision_path,
            "forecast"
        )
    ).compute().reset_index(drop=True)
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text="Insample and Blind Forecasts")
        )
    )
    test_fd_retail_min["experiment_definition"]['time_horizons'] = [0]
    test_fd_retail_min["experiment_definition"]['confidence_intervals'] = []
    for h in test_fd_retail_min["experiment_definition"]['time_horizons']:
        result_df_2d = result_df.sort_values('Promo').groupby('Date').agg('first').reset_index()

        if len(test_fd_retail_min["experiment_definition"]['confidence_intervals']) > 0:
            for i in test_fd_retail_min["experiment_definition"]['confidence_intervals']:
                fig.add_trace(go.Scatter(marker=dict(color="cyan"), mode="lines",
                                         x=result_df_2d[test_fd_retail_min["experiment_definition"]['time_index']],
                                         y=result_df_2d[
                                             '{}_h_{}_pred_c_{}'.format(
                                                 test_fd_retail_min["experiment_definition"]['target'],
                                                 h, i)], name="h_{}_c_{}".format(h, i)))
            fig.update_traces(fill='tonexty', name='Confidence Bound', selector=dict(
                name="h_{}_c_{}".format(h, test_fd_retail_min["experiment_definition"]['confidence_intervals'][-1])))
            fig.update_traces(showlegend=False, name='Upper Confidence Bound', selector=dict(
                name="h_{}_c_{}".format(h, test_fd_retail_min["experiment_definition"]['confidence_intervals'][0])))
        fig.add_trace(
            go.Scatter(marker=dict(color='black'), line=dict(dash='dash'), mode="lines",
                           x=result_df_2d[test_fd_retail_min["experiment_definition"]['time_index']],
                           y=result_df_2d[test_fd_retail_min["experiment_definition"]['target']],
                           name=test_fd_retail_min["experiment_definition"]['target']))
        fig.add_trace(go.Scatter(marker=dict(color="darkblue"), mode="lines",
                                     x=result_df_2d[test_fd_retail_min["experiment_definition"]['time_index']],
                                     y=result_df_2d[
                                         '{}_h_{}_pred'.format(test_fd_retail_min["experiment_definition"]['target'],
                                                               h)], name="Horizon {} Forecast".format(h)))
        path = pathlib.Path(pathlib.Path(__file__).parent.parent.parent,
                                    'docs_src/plots/retail_min/h_{}_2d.html'.format(h))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path)
    ###TODO START HERE. GET FACTOR PLOT READY AND IMPLEMENT ASSERTION
    '''fig = go.Figure()
    for h in test_fd_retail_min["experiment_definition"]['time_horizons']:
        result_df_2d = result_df[(result_df['Promo'] == 1) & (result_df['Store_1.0'] == 1) & (result_df['Date'] >= '2015-08-01')]
        factors = [c for c in result_df_2d if c.split("_")[0] == "factor"]
        result_df_2d = result_df_2d[factors + [test_fd_retail_min["experiment_definition"]['time_index']]]
        for f in factors:
            fig.add_trace(go.Bar(x=result_df_2d[test_fd_retail_min["experiment_definition"]['time_index']],
                                 y=result_df_2d[
                                     f], name="_".join(f.split("_")[1:])))
            fig.update_layout(barmode='relative', title_text='Factor')
        fig.write_html('docs_src/plots/test_forecast_retail_h_{}_factors.html'.format(h))
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text="Forecasted Promo Impact"),
            scene=dict(
                xaxis=dict(
                    title='Date'),
                yaxis=dict(
                    title='Promo', tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes']),
                zaxis=dict(
                    title='Sales'))
        )
    )'''
