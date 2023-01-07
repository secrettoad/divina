from divina.divina.pipeline.pipeline import Pipeline

quickstart_pipeline_2 = Pipeline(target="Sales",
    causal_model_params={"link_function":"log"},
    target_dimensions=[
      "Store"
    ],
    time_index="Date",
    frequency="S")