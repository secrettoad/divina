from divina import Divina

quickstart_pipeline_5 = Divina(
    target="Sales",
    causal_model_params=[{"link_function": "log"}, {"link_function": "None"}],
    target_dimensions=["Store"],
    time_index="Date",
    frequency="D",
    drop_features=[
        "Customers",
        "StoreType",
        "Assortment",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
        "Weekday",
        "Month",
        "Holiday",
        "Year",
    ],
    time_features=True,
    encode_features=[
        "Store",
        "Month",
        "StoreType",
        "Weekday",
        "HolidayType",
    ],
    bin_features={"Month": [3, 6, 9]},
    interaction_features={"Store": ["HolidayType"]},
    validation_splits=["2014-06-01"],
)
