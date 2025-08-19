# AeroClub RecSys 2025 - XGBoost Ranking

# This notebook implements an improved ranking approach using XGBoost and Polars for the AeroClub recommendation challenge.

# Based on AeroClub RecSys 2025 - XGBoost Ranking Baseline from Kirill Khoruzhii
# https://www.kaggle.com/code/ka1242/xgboost-ranker-with-polars


from model import XGB_Ensemble

model = XGB_Ensemble()
model.train()

