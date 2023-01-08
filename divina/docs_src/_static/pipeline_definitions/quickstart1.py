from divina.divina.pipeline.pipeline import Pipeline

quickstart_pipeline_1 = Pipeline(target="Sales",
    causal_model_params=[{"link_function": "log", "linear_parameters":{"penalty":"l1"}}, {"link_function": "None", "linear_parameters":{"penalty":"l1"}}
                        ,{"link_function": "log", "linear_parameters":{"penalty":"l2"}}, {"link_function": "None", "linear_parameters":{"penalty":"l2"}}],

    time_index="Date",
    frequency="D",
    drop_features=['Customers', 'StoreType', 'Assortment', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval'])


