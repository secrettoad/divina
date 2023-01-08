from divina.divina.pipeline.pipeline import Pipeline

quickstart_pipeline_3 = Pipeline(
    target="Sales",
    causal_model_params=[{"link_function": "log"}, {"link_function": "None"}],
    target_dimensions=[
      "Store"
    ],
    time_index="Date",
    drop_features=['Customers', 'StoreType', 'Assortment', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval',
                   'Weekday', 'Month', 'Holiday', 'Year'],
    frequency="D",
    time_features=True
)