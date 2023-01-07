from divina.divina.pipeline.pipeline import Pipeline

###TODO - add time dataset to retail dataset
quickstart_pipeline_5 = Pipeline(
    target="Sales",
    causal_model_params={"link_function": "log"},
    target_dimensions=[
      "Store"
    ],
    time_index="Date",
    frequency="S",
    include_features=[
      "Store",
      "Weekday",
      "Month",
      "Holiday",
      "HolidayType",
      "StoreType",
      "Assortment",
      "LastDayOfMonth"
    ],
    encode_features=[
      "Store",
      "Month",
      "StoreType",
      "Weekday",
      "HolidayType",
      "Assortment"
    ],
    bin_features={
      "Month": [
        3,
        6,
        9
      ]
    },
    interaction_features={
      "Store": [
        "Holiday"
      ]
    },
    validation_splits=[
      "2015-07-18"
    ]
)