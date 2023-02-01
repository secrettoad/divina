import pathlib

import dask.dataframe as ddf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from divina.divina.datasets.load import _load
from divina.divina.pipeline.pipeline import (
    assert_pipeline_fit_result_equal,
    assert_pipeline_predict_result_equal,
)


def test_preprocess(
    test_data_1,
    test_df_1,
    test_pipeline_1,
):
    df = test_pipeline_1.preprocess(test_data_1)
    pd.testing.assert_frame_equal(
        df.compute(),
        test_df_1.compute(),
    )


def test_train(
    test_df_1,
    test_pipeline_1,
    test_model_1,
    random_state,
):
    model = test_pipeline_1.train(
        x=test_df_1.drop(columns="c"),
        y=test_df_1["c"],
        random_state=random_state,
        model_type="GLM",
        model_params={"link_function": "log"},
    )
    assert model == test_model_1


def test_forecast(
    test_df_1,
    test_pipeline_1,
    test_model_1,
    test_forecast_1,
):
    result = test_pipeline_1.forecast(
        model=test_model_1, x=test_df_1.set_index("a").drop(columns="c")
    )
    np.testing.assert_allclose(
        result.compute().values, test_forecast_1.compute().values
    )


def test_validate(test_pipeline_1, test_df_1, test_forecast_1, test_metrics_1):
    metrics = test_pipeline_1.validate(
        truth_dataset=test_df_1["c"], prediction_dataset=test_forecast_1
    )
    assert metrics == test_metrics_1


def test_pipeline_fit(test_data_1, test_pipeline_2, test_pipeline_fit_result):
    result = test_pipeline_2.fit(test_data_1)
    assert_pipeline_fit_result_equal(result, test_pipeline_fit_result)


def test_pipeline_predict(
    test_data_1,
    test_pipeline_2,
    test_horizons,
    test_scenarios,
    test_simulate_end,
    test_simulate_start,
    test_bootstrap_models,
    test_boost_models,
    test_pipeline_predict_result,
):
    test_pipeline_2.is_fit = True
    test_pipeline_2.bootstrap_models = test_bootstrap_models
    test_pipeline_2.boost_models = test_boost_models
    result = test_pipeline_2.predict(
        x=test_data_1[test_data_1["a"] >= "1970-01-01 00:00:05"],
        boost_y=test_pipeline_2.target,
        horizons=test_horizons,
    )
    assert_pipeline_predict_result_equal(result, test_pipeline_predict_result)


def test_quickstart(test_pipelines_quickstart, random_state):
    for i, pipeline in enumerate(test_pipelines_quickstart):
        print("testing quickstart {}".format(i))
        result = pipeline.fit(
            _load("divina://retail_sales"),
            start="2013-01-01 00:00:00",
            end="2015-03-31 00:00:00",
        )
        result_df = result[0].truth
        factors = result[0].causal_validation.factors
        if factors is not None:
            for c in factors:
                result_df[c] = factors[c]
        result_df["y_hat"] = result[0].causal_validation.predictions

        if result_df.index.name == "__target_dimension_index__":
            result_df = pipeline.extract_dask_multiindex(result_df)
        else:
            result_df = result_df.reset_index()
        if i in [6, 7, 8]:
            x = _load("divina://retail_sales")
            x = x[
                (x["Date"] >= "2015-04-01 00:00:00")
                & (x["Date"] < "2015-08-01 00:00:00")
            ]
            if i == 8:
                predict_result = pipeline.predict(
                    x, horizons=pipeline.time_horizons, boost_y=pipeline.target
                )
            else:
                predict_result = pipeline.predict(
                    x.drop(columns=pipeline.target),
                    horizons=pipeline.time_horizons,
                )
            predict_result_df = predict_result.truth
            x["Store"] = x["Store"].astype(float)
            y = pipeline.set_dask_multiindex(
                x[[pipeline.target, pipeline.time_index] + pipeline.target_dimensions]
            )
            predict_result_df[pipeline.target] = y[pipeline.target]
            predict_result_df["y_hat"] = predict_result.causal_predictions.predictions
            if i == 8:
                for h in pipeline.time_horizons:
                    predict_result_df["y_hat_h_{}".format(h)] = predict_result[
                        h
                    ].predictions
                    confidence_intervals = predict_result[h].confidence_intervals
                    if confidence_intervals is not None:
                        for _c in confidence_intervals:
                            predict_result_df[
                                _c.replace("Sales_pred", "y_hat_h_{}".format(h))
                            ] = confidence_intervals[_c]
            confidence_intervals = (
                predict_result.causal_predictions.confidence_intervals
            )
            if confidence_intervals is not None:
                for _c in confidence_intervals:
                    predict_result_df[
                        _c.replace("Sales_pred", "y_hat")
                    ] = confidence_intervals[_c]
            factors = predict_result.causal_predictions.factors
            if factors is not None:
                for c in factors:
                    predict_result_df[c] = factors[c]
            result_df = pipeline.extract_dask_multiindex(
                result_df.append(predict_result_df)
            )
            result_df.Store = ddf.to_numeric(result_df.Store).astype(int)

        result_df = result_df.compute()
        if not pipeline.target_dimensions:
            stores = [2]
        else:
            stores = [1, 2, 3]
        result_df = result_df[result_df["Date"] >= "2015-01-01"]
        if i in [0, 1, 2, 3, 4, 5]:
            result_df = result_df[result_df["Date"] < "2017-08-01"]
        for s in stores:
            fig = go.Figure()
            if i in [0, 1, 2, 3]:
                store_df = result_df[result_df["Store"] == s]
            else:
                store_df = result_df[result_df["Store_{}".format(float(s))] == 1]
            if pipeline.scenarios:
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
            if pipeline.confidence_intervals:
                if len(pipeline.confidence_intervals) > 0:
                    for _c in pipeline.confidence_intervals:
                        fig.add_trace(
                            go.Scatter(
                                marker=dict(color="cyan"),
                                mode="lines",
                                x=store_df[pipeline.time_index],
                                y=store_df["y_hat_c_{}".format(_c)],
                                name="y_hat_c_{}".format(_c),
                            )
                        )
                    fig.update_traces(
                        fill="tonexty",
                        name="Confidence Bound",
                        selector=dict(
                            name="y_hat_c_{}".format(
                                pipeline.confidence_intervals[-1],
                            )
                        ),
                    )
                    fig.update_traces(
                        showlegend=False,
                        name="Upper Confidence Bound",
                        selector=dict(
                            name="y_hat_c_{}".format(
                                pipeline.confidence_intervals[0],
                            )
                        ),
                    )
            fig.add_trace(
                go.Scatter(
                    marker=dict(color="black"),
                    line=dict(dash="dash"),
                    mode="lines",
                    x=store_df[pipeline.time_index],
                    y=store_df[pipeline.target],
                    name=pipeline.target,
                )
            )
            fig.add_trace(
                go.Scatter(
                    marker=dict(color="darkblue"),
                    mode="lines",
                    x=store_df[pipeline.time_index],
                    y=store_df["y_hat"],
                    name="Forecast",
                )
            )
            for h, t in zip(pipeline.time_horizons, ["red", "green", "purple"]):
                fig.add_trace(
                    go.Scatter(
                        marker=dict(color=t),
                        mode="lines",
                        x=store_df[pipeline.time_index],
                        y=store_df["y_hat_h_{}".format(h)],
                        name="Boost Horizon {}".format(h),
                    )
                )
            path = pathlib.Path(
                pathlib.Path(__file__).parent.parent.parent,
                "docs_heavy_assets/_static/plots/"
                "quickstart/{}_s_{}_2d.html".format(i, s),
            )
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.02,
                    xanchor="center",
                    x=0.5,
                )
            )
            fig.update_xaxes(side="top")
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(path)
            factor_fig = go.Figure()
            _factors = [c for c in store_df if c.split("_")[0] == "factor"]
            if pipeline.scenarios:
                store_df = store_df[
                    (store_df["Date"] > "2015-08-01") | (result_df["Promo"] == 1)
                ]
            store_df = store_df[_factors + [pipeline.time_index]]
            for f in _factors:
                if f.split("_")[1] in [
                    "Open",
                    "CompetitionOpenSinceMonth",
                    "CompetitionOpenSinceYear",
                    "T",
                ]:
                    visible = "legendonly"
                else:
                    visible = None
                factor_fig.add_trace(
                    go.Bar(
                        x=store_df[pipeline.time_index],
                        y=store_df[f],
                        name=("_".join(f.split("_")[1:])[:15] + "..")
                        if len("_".join(f.split("_")[1:])) > 17
                        else "_".join(f.split("_")[1:]),
                        visible=visible,
                    )
                )
            factor_fig.update_layout(
                barmode="relative",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.05,
                    xanchor="center",
                    x=0.5,
                ),
            )
            factor_fig.update_xaxes(side="top")
            path = pathlib.Path(
                pathlib.Path(__file__).parent.parent.parent,
                "docs_heavy_assets/plots/quickstart/"
                "{}_test_forecast_retail_s_{}_factors.html".format(i, s),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            factor_fig.write_html(path)
