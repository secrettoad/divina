from divina.divina.pipeline.pipeline import Pipeline

quickstart_pipeline_1 = Pipeline(target="Sales",
    causal_model_params={"link_function": "log"},
    time_index="Date",
    frequency="S")

