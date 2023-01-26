from divina import Divina

quickstart_pipeline_0 = Divina(
    target="Sales",
    time_index="Date",
    frequency="D",
    drop_features=[
        "Customers",
        "StoreType",
        "Assortment",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
    ],
)
