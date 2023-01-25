from divina import Divina

quickstart_pipeline_1 = Divina(target="Sales",
    causal_model_params=[{"link_function": "log"}, {"link_function": "None"}],
    time_index="Date",
    frequency="D",
    drop_features=['Customers', 'StoreType', 'Assortment', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval'])


