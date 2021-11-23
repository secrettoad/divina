import os
import json
from ..train import _train
from ..forecast import _forecast
from ..dataset import _get_dataset
from ..validate import _validate
from ..experiment import _experiment
import pathlib
from ..utils import compare_sk_models, get_parameters, set_parameters
import joblib
import pandas as pd
import dask.dataframe as ddf
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import plotly.graph_objects as go


def test_validate_experiment_definition(
        ed_no_target,
        ed_time_horizons_not_list,
        ed_time_validation_splits_not_list,
        ed_no_time_index,
        ed_no_data_path,
        ed_invalid_model,
        ed_time_horizons_range_not_tuple
):
    for dd in [
        ed_no_target,
        ed_no_time_index,
        ed_time_validation_splits_not_list,
        ed_time_horizons_not_list,
        ed_no_data_path,
        ed_invalid_model,
        ed_time_horizons_range_not_tuple
    ]:
        try:
            with open(pathlib.Path(pathlib.Path(__file__).parent.parent, 'config/ed_schema.json'), 'r') as f:
                validate(instance=dd, schema=json.load(f))
        except ValidationError:
            return None
        else:
            assert False


def test_get_composite_dataset(
        test_df_1,
        test_df_2,
        test_ed_2,
        test_composite_dataset_1,
        dask_client,
):
    for dataset in test_ed_2["experiment_definition"]["joins"]:
        pathlib.Path(
            dataset["data_path"]
        ).mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        test_ed_2["experiment_definition"]["data_path"],
    ).mkdir(parents=True, exist_ok=True)

    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_ed_2["experiment_definition"]["data_path"]
    )
    ddf.from_pandas(test_df_2, npartitions=2).to_parquet(

        test_ed_2["experiment_definition"]["joins"][0]["data_path"]
    )
    df = _get_dataset(test_ed_2["experiment_definition"])

    pd.testing.assert_frame_equal(df.compute().reset_index(drop=True), test_composite_dataset_1.reset_index(drop=True))


def test_train(s3_fs, test_df_1, test_ed_1, test_model_1, dask_client, test_bootstrap_models, test_validation_models,
               random_state):
    experiment_path = "divina-test/experiment/test1"
    pathlib.Path(
        test_ed_1["experiment_definition"]["data_path"],
    ).mkdir(parents=True, exist_ok=True)
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_ed_1["experiment_definition"]["data_path"]
    )
    _train(
        s3_fs=s3_fs,
        experiment_definition=test_ed_1["experiment_definition"],
        write_path=experiment_path,
        random_state=random_state,
    )

    compare_sk_models(
        joblib.load(
            os.path.abspath(
                os.path.join(
                    experiment_path,
                    "models",
                    "h-1",
                )
            )
        ),
        test_model_1[0],
    )
    with open(os.path.abspath(
            os.path.join(
                experiment_path,
                "models",
                "h-1_params.json",
            )
    )) as f:
        assert json.load(f) == test_model_1[1]
    for state in test_bootstrap_models:
        compare_sk_models(
            joblib.load(
                os.path.abspath(
                    os.path.join(
                        experiment_path,
                        "models/bootstrap",
                        "h-1_r-{}".format(state),
                    )
                )
            ),
            test_bootstrap_models[state][0],
        )
        with open(os.path.abspath(
                os.path.join(
                    experiment_path,
                    "models/bootstrap",
                    "h-1_r-{}_params.json".format(state),
                )
        )) as f:
            assert json.load(f) == test_bootstrap_models[state][1]
    for split in test_validation_models:
        compare_sk_models(
            joblib.load(
                os.path.abspath(
                    os.path.join(
                        experiment_path,
                        "models",
                        "s-{}_h-1".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
                    )
                )
            ),
            test_validation_models[split][0],
        )
        with open(os.path.abspath(
                os.path.join(
                    experiment_path,
                    "models",
                    "s-{}_h-1_params.json".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
                )
        )) as f:
            features = json.load(f)
        assert features == test_validation_models[split][1]


def test_forecast(
        s3_fs, dask_client, test_df_1, test_ed_1, test_model_1, test_val_predictions_1, test_forecast_1,
        test_bootstrap_models
):
    experiment_path = "divina-test/experiment/test1"
    pathlib.Path(
        test_ed_1["experiment_definition"]["data_path"]
    ).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(experiment_path, "models/bootstrap")).mkdir(
        parents=True, exist_ok=True
    )
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_ed_1["experiment_definition"]["data_path"]
    )
    joblib.dump(
        test_model_1[0],
        os.path.join(
            experiment_path,
            "models",
            "h-1",
        ),
    )
    with open(os.path.join(
            experiment_path,
            "models",
            "h-1_params.json",
    ), 'w+') as f:
        json.dump(
            test_model_1[1],
            f
        )
    for state in test_bootstrap_models:
        joblib.dump(
            test_bootstrap_models[state][0],
            os.path.join(
                experiment_path,
                "models/bootstrap",
                "h-1_r-{}".format(state),
            ),
        )
        with open(os.path.join(
                experiment_path,
                "models/bootstrap",
                "h-1_r-{}_params.json".format(state),
        ), 'w+') as f:
            json.dump(
                test_bootstrap_models[state][1],
                f
            )

    _forecast(
        s3_fs=s3_fs,
        experiment_definition=test_ed_1["experiment_definition"],
        read_path=experiment_path,
        write_path=experiment_path,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                experiment_path,
                "forecast"
            )
        ).compute().reset_index(drop=True),
        test_forecast_1.reset_index(drop=True), check_dtype=False
    )


def test_validate(
        s3_fs, test_ed_1, test_df_1, test_metrics_1, dask_client, test_val_predictions_1, test_validation_models,
        test_model_1,
        test_bootstrap_models
):
    experiment_path = "divina-test/experiment/test1"
    ddf.from_pandas(test_df_1, npartitions=2).to_parquet(
        test_ed_1["experiment_definition"]["data_path"]
    )
    pathlib.Path(
        os.path.join(
            experiment_path, "models", "bootstrap"
        )
    ).mkdir(parents=True, exist_ok=True)

    for split in test_validation_models:
        joblib.dump(
            test_validation_models[split][0],
            os.path.join(
                experiment_path,
                "models",
                "s-{}_h-1".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
            ),
        )
        with open(os.path.join(
                experiment_path,
                "models",
                "s-{}_h-1_params.json".format(pd.to_datetime(str(split)).strftime("%Y%m%d-%H%M%S")),
        ), 'w+') as f:
            json.dump(
                test_validation_models[split][1],
                f
            )
    for state in test_bootstrap_models:
        joblib.dump(
            test_bootstrap_models[state][0],
            os.path.join(
                experiment_path,
                "models/bootstrap",
                "h-1_r-{}".format(state),
            ),
        )
        with open(os.path.join(
                experiment_path,
                "models/bootstrap",
                "h-1_r-{}_params.json".format(state),
        ), 'w+') as f:
            json.dump(
                test_bootstrap_models[state][1],
                f
            )
    _validate(
        s3_fs=s3_fs,
        experiment_definition=test_ed_1["experiment_definition"],
        read_path=experiment_path,
        write_path=experiment_path,
    )
    pd.testing.assert_frame_equal(
        ddf.read_parquet(
            os.path.join(
                experiment_path,
                "validation",
                "s-19700101-000007",
            )
        ).compute().reset_index(drop=True),
        test_val_predictions_1.reset_index(drop=True), check_dtype=False
    )
    with open(
            os.path.join(experiment_path, "metrics.json"),
            "r",
    ) as f:
        assert json.load(f) == test_metrics_1


def test_get_params(
        s3_fs, test_model_1, test_params_1
):
    experiment_path = "divina-test/experiment/test1"
    pathlib.Path(os.path.join(experiment_path, "models")).mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(
        test_model_1,
        os.path.join(
            experiment_path,
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    with open(os.path.join(
            experiment_path,
            "models",
            "s-19700101-000007_h-1_params",
    ), 'w+') as f:
        json.dump(test_model_1[1], f)
    params = get_parameters(s3_fs=s3_fs, model_path=os.path.join(
        experiment_path,
        "models",
        "s-19700101-000007_h-1",
    ))

    assert params == test_params_1


def test_set_params(
        s3_fs, test_model_1, test_params_1, test_params_2
):
    experiment_path = "divina-test/experiment/test1"
    pathlib.Path(os.path.join(experiment_path, "models")).mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(
        test_model_1,
        os.path.join(
            experiment_path,
            "models",
            "s-19700101-000007_h-1",
        ),
    )
    with open(os.path.join(
            experiment_path,
            "models",
            "s-19700101-000007_h-1_params",
    ), 'w+') as f:
        json.dump(test_params_1, f)
    set_parameters(s3_fs=s3_fs, model_path=os.path.join(
        experiment_path,
        "models",
        "s-19700101-000007_h-1",
    ), params=test_params_2['features'])

    with open(os.path.join(
            experiment_path,
            "models",
            "s-19700101-000007_h-1_params",
    ), 'rb') as f:
        params = json.load(f)

    assert params == test_params_2


def test_quickstart(test_eds_quickstart, random_state):
    for k in test_eds_quickstart:
        ed = test_eds_quickstart[k]
        experiment_path = "divina-test/experiment/test1"
        _experiment(
            experiment_definition=ed["experiment_definition"],
            read_path=experiment_path,
            write_path=experiment_path,
            random_state=11
        )
        result_df = ddf.read_parquet(
            os.path.join(
                experiment_path,
                "forecast"
            )
        ).compute().reset_index(drop=True)
        ###RESET
        '''ddf.read_parquet(
            os.path.join(
                experiment_path,
                "forecast"
            )
        ).to_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'docs_src/results/forecasts',
                               k))'''
        pd.testing.assert_frame_equal(result_df, ddf.read_parquet(pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'docs_src/results/forecasts',
                               k)).compute().reset_index(drop=True))
        ed["experiment_definition"]['time_horizons'] = [0]
        if not "target_dimensions" in ed["experiment_definition"]:
            stores = [6]
        else:
            stores = [1, 2, 3]
        result_df = result_df[result_df['Date'] >= '2015-01-01']
        for s in stores:
            fig = go.Figure()
            for h in ed["experiment_definition"]['time_horizons']:
                if not "encode_features" in ed["experiment_definition"]:
                    store_df = result_df[result_df['Store'] == s]
                else:
                    store_df = result_df[result_df['Store_{}'.format(float(s))] == 1]
                if "scenarios" in ed["experiment_definition"]:
                    store_df = store_df[(store_df['Date'] < "2015-08-01") | (result_df['Promo'] == 1)]
                    fig.add_vrect(x0=pd.to_datetime('07-31-2015').timestamp() * 1000,
                                  x1=pd.to_datetime('01-01-2016').timestamp() * 1000, line_width=2,
                                  line_color="cadetblue",
                                  annotation_text='Forecasts assuming promotions')
                if "confidence_intervals" in ed["experiment_definition"]:
                    if len(ed["experiment_definition"]['confidence_intervals']) > 0:
                        for i in ed["experiment_definition"]['confidence_intervals']:
                            fig.add_trace(go.Scatter(marker=dict(color="cyan"), mode="lines",
                                                     x=store_df[ed["experiment_definition"]['time_index']],
                                                     y=store_df[
                                                         '{}_h_{}_pred_c_{}'.format(
                                                             ed["experiment_definition"]['target'],
                                                             h, i)], name="h_{}_c_{}".format(h, i)))
                        fig.update_traces(fill='tonexty', name='Confidence Bound', selector=dict(
                            name="h_{}_c_{}".format(h, ed["experiment_definition"]['confidence_intervals'][-1])))
                        fig.update_traces(showlegend=False, name='Upper Confidence Bound', selector=dict(
                            name="h_{}_c_{}".format(h, ed["experiment_definition"]['confidence_intervals'][0])))
                fig.add_trace(
                        go.Scatter(marker=dict(color='black'), line=dict(dash='dash'), mode="lines",
                                       x=store_df[ed["experiment_definition"]['time_index']],
                                       y=store_df[ed["experiment_definition"]['target']],
                                       name=ed["experiment_definition"]['target']))
                fig.add_trace(go.Scatter(marker=dict(color="darkblue"), mode="lines",
                                             x=store_df[ed["experiment_definition"]['time_index']],
                                             y=store_df[
                                                 '{}_h_{}_pred'.format(ed["experiment_definition"]['target'],
                                                                       h)], name="Forecast".format(h)))
                path = pathlib.Path(pathlib.Path(__file__).parent.parent.parent,
                                            'docs_src/_static/plots/quickstart/{}_h_{}_s_{}_2d.html'.format(k, h, s))
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(path)
                factor_fig = go.Figure()
                factors = [c for c in store_df if c.split("_")[0] == "factor"]
                if "scenarios" in ed["experiment_definition"]:
                    store_df = store_df[(store_df['Date'] > "2015-08-01") | (result_df['Promo'] == 1)]
                store_df = store_df[factors + [ed["experiment_definition"]['time_index']]]
                for f in factors:
                    factor_fig.add_trace(go.Bar(x=store_df[ed["experiment_definition"]['time_index']],
                                         y=store_df[
                                             f], name=("_".join(f.split("_")[1:])[:15] + '..') if len("_".join(f.split("_")[1:])) > 17 else "_".join(f.split("_")[1:])))
                factor_fig.update_layout(barmode='relative')
                path = pathlib.Path(pathlib.Path(__file__).parent.parent.parent,
                                    'docs_src/_static/plots/quickstart/{}_test_forecast_retail_h_{}_s_{}_factors.html'.format(k, h, s))
                path.parent.mkdir(parents=True, exist_ok=True)
                factor_fig.write_html(path)


