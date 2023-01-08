from divina.divina.pipeline.pipeline import Pipeline

quickstart_pipeline_0 = Pipeline(target="Sales",
    time_index="Date",
    frequency="D",
    drop_features=['Customers', 'StoreType', 'Assortment', 'Promo2SinceWeek',
                   'Promo2SinceYear', 'PromoInterval'])


