# AeroClub RecSys 2025 - XGBoost Ranking

# This notebook implements an improved ranking approach using XGBoost and Polars for the AeroClub recommendation challenge.

# Based on AeroClub RecSys 2025 - XGBoost Ranking Baseline from Kirill Khoruzhii
# https://www.kaggle.com/code/ka1242/xgboost-ranker-with-polars

import polars as pl

from common import Cfg
from model import XGB_Ensemble


model = XGB_Ensemble()
ensemble_preds = model.predict(Cfg.test_path)

test = pl.read_parquet(Cfg.test_path).drop('__index_level_0__').with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))

# Без постобработки
submission_xgb = (
    test.select(['Id', 'ranker_id'])

    .with_columns(pl.Series('pred_score', ensemble_preds))
    .with_columns(
        pl.col('pred_score')
        .rank(method='ordinal', descending=True)
        .over('ranker_id')
        .cast(pl.Int32)
        .alias('selected')
    )
    .select(['Id', 'ranker_id', 'selected'])
)
print("Сохраняем сабмит")
submission_xgb.write_csv(Cfg.submissions_path)
