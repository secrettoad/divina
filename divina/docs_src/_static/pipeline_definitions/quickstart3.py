from divina.divina.pipeline.pipeline import Pipeline

###TODO - add time dataset to retail dataset
quickstart_pipeline_3 = Pipeline(
    target="Sales",
    causal_model_params={"link_function": "log"},
    target_dimensions=[
      "Store"
    ],
    time_index="Date",
    frequency="S"
)