# AeroClub RecSys 2025 - XGBoost Ranking

# This notebook implements an improved ranking approach using XGBoost and Polars for the AeroClub recommendation challenge.

# Based on AeroClub RecSys 2025 - XGBoost Ranking Baseline from Kirill Khoruzhii
# https://www.kaggle.com/code/ka1242/xgboost-ranker-with-polars

import polars as pl
import xgboost as xgb
import numpy as np

import random
from copy import deepcopy
import pickle
from datetime import datetime

from common import Cfg
from prepare_data import get_dmatrix


class XGB_Ensemble:
    def __init__(self):
        
        # Список в котором будут отрренированааные модели
        self.models = []
        # Датасет с признаками, подготовленными к обучению
        self.train_source = None

    
    ## Helpers
    def hitrate_at_3(self, y_true, y_pred, groups):
        
        df = pl.DataFrame({
            'group': groups,
            'pred': y_pred,
            'true': y_true
        })
        
        return (
            df.filter(pl.col("group").count().over("group") > 10)
            .sort(["group", "pred"], descending=[False, True])
            .group_by("group", maintain_order=True)
            .head(3)
            .group_by("group")
            .agg(pl.col("true").max())
            .select(pl.col("true").mean())
            .item()
        )


    def make_short_groups(self, max_per_group=50):
        '''Делает короткие группы из тех что превышают заданный размер'''

        train = self.train_source.clone()

        rng = np.random.default_rng(Cfg.RANDOM_STATE)

        # Получаем уникальные ranker_id
        unique_ids = train.select("ranker_id").unique()
        shifts = rng.integers(-2, max_per_group//5, size=unique_ids.height)  # от -2 до 30 включительно

        # Создаем таблицу с ranker_id и соответствующим сдвигом
        shift_table = unique_ids.with_columns([
            pl.Series("group_shift", shifts)
        ])

        # Добавляем в train индекс строки и объединяем с shift_table по ranker_id
        rand_series = pl.Series("rand", rng.random(len(train)))
        print(train.height)
        train = (
            train
            .with_row_index("row_idx")
            .with_columns(rand_series)
            .join(shift_table, on="ranker_id", how="left")  # добавляем group_shift
            .with_columns([
                pl.len().over("ranker_id").alias("grp_size"),
                pl.col("rand").rank(method="dense").over("ranker_id").alias("rand_rank"),
                (max_per_group + pl.col("group_shift")).alias("adjusted_max")
            ])
            .filter(
                (pl.col("grp_size") <= max_per_group) |             # маленькие группы — целиком
                (pl.col("selected") == 1) |                         # выбранные строки всегда
                (pl.col("rand_rank") <= pl.col("adjusted_max"))     # большие группы — отбираем с учетом смещения
            )
            .sort("row_idx")
            .drop(["grp_size", "rand", "rand_rank", "row_idx", "adjusted_max", "group_shift"])
        )
        print(train.height)

        return train


    def ensure_1pct_minus1(self, df: pl.DataFrame) -> pl.DataFrame:
        '''Добавляет 1% пропусков случайно. Пропуски это -1.'''
        
        df_new = df.clone()

        n_rows = df.height
        target_count = int(np.floor(n_rows * 0.005))  # 1% от строк

        for col in df.columns:
            series = df_new[col]
            
            # Пропускаем булевые колонки
            if series.dtype == pl.Boolean:
                continue

            current_minus1_count = (series == -1).sum()

            if current_minus1_count >= target_count:
                # Уже есть ≥1% -1, пропускаем
                continue

            # Сколько ещё нужно добавить -1
            need_to_replace = target_count - current_minus1_count
            if need_to_replace <= 0:
                continue

            # Индексы, которые не равны -1
            available_indices = np.where(series.to_numpy() != -1)[0]

            # Случайные индексы для замены
            replace_indices = np.random.choice(
                available_indices, size=need_to_replace, replace=False
            )

            # Создаём копию колонки с заменой
            col_values = series.to_numpy().copy()
            col_values[replace_indices] = -1

            # Обновляем колонку
            df_new = df_new.with_columns(pl.Series(name=col, values=col_values))

        return df_new
    

    def train(self):

        self.train_source = pl.read_parquet(Cfg.train_path_processed)

        if Cfg.IS_VALIDATE:
            validate = pl.read_parquet(Cfg.valid_path_processed)
            y_va = validate.select('selected')
            groups_va = validate.select('ranker_id')
            validate = validate.drop(["selected", "ranker_id"])
            group_sizes_va = groups_va.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()
            dval = xgb.DMatrix(validate, label=y_va, missing=-1, group=group_sizes_va, feature_names=list(validate.columns))


        all_preds = []  # Список для хранения предсказаний каждой модели

        base_params_1 = {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg@3',
            'learning_rate': 0.03430629350470104,
            'max_depth': 20,
            'min_child_weight': 24,
            'subsample': 0.9680833434687813,
            'colsample_bytree': 0.1859969541434444,
            'lambda': 13.73979976725866,
            'alpha': 4.315374828140574,
            'gamma': 0.145064313893779,
            # 'n_jobs': 48,
            'n_jobs': -1,
        }

        base_params_2 = {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg@3',
            'learning_rate': 0.01874682748459287,
            'max_depth': 17,
            'min_child_weight': 14,
            'subsample': 0.9910104516963302,
            'colsample_bytree': 0.24822082790651878,
            'lambda': 14.476928639040858,
            'alpha': 0.06344642592268934,
            'gamma': 0.2556097003945161,
            # 'n_jobs': 48,
            'n_jobs': -1,
        }

        # Список различных комбинаций параметров
        params_list = [
            {**deepcopy(base_params_2), 'num_boost_round': Cfg.num_boost_round},  # Базовая модель
            {**deepcopy(base_params_1), 'num_boost_round': Cfg.num_boost_round+50, 'objective': 'rank:ndcg', 'seed': Cfg.RANDOM_STATE+1},
            {**deepcopy(base_params_2), 'num_boost_round': Cfg.num_boost_round+100, 'seed': Cfg.RANDOM_STATE+2},  # Базовая модель
            {**deepcopy(base_params_1), 'num_boost_round': Cfg.num_boost_round-50, 'seed': Cfg.RANDOM_STATE+3},
            {**deepcopy(base_params_2), 'num_boost_round': Cfg.num_boost_round+500, 'seed': Cfg.RANDOM_STATE+4},

            
            {**deepcopy(base_params_2), 'num_boost_round': Cfg.num_boost_round, 'seed': Cfg.RANDOM_STATE+5},  # Базовая модель
            {**deepcopy(base_params_1), 'num_boost_round': Cfg.num_boost_round+50, 'objective': 'rank:ndcg', 'seed': Cfg.RANDOM_STATE+6},
            {**deepcopy(base_params_2), 'num_boost_round': Cfg.num_boost_round+100, 'seed': Cfg.RANDOM_STATE+7},  # Базовая модель
            {**deepcopy(base_params_1), 'num_boost_round': Cfg.num_boost_round-50, 'seed': Cfg.RANDOM_STATE+8},
            {**deepcopy(base_params_2), 'num_boost_round': Cfg.num_boost_round+500, 'seed': Cfg.RANDOM_STATE+9},
        ]

        # Список для хранения обученных моделей
        trained_models_xgb = []

        MAX_PER_GROUP = 50

        # Обучение каждой модели
        for params in params_list:
            print(f"\nTraining model with params: {params}")

            train = self.make_short_groups(max_per_group=MAX_PER_GROUP)
            
            y_tr = train.select('selected')
            groups_tr = train.select('ranker_id')
            train = train.drop(["selected", "ranker_id", "requestDate", "profileId"])


            print('Add -1 to train')
            # Добавляем -1 до 1% в данных (-1 это null)
            train = self.ensure_1pct_minus1(train)

            group_sizes_tr = groups_tr.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()
            dtrain = xgb.DMatrix(train, label=y_tr, missing=-1, group=group_sizes_tr, feature_names=list(train.columns))

            evals = [(dtrain, 'train'), (dval, 'val')] if Cfg.IS_VALIDATE else None
            
            # Извлекаем num_boost_round, так как он не является параметром XGBoost, а передается отдельно
            num_boost_round = params.pop('num_boost_round')
            
            print('Start train model', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            # Обучаем модель
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                # num_boost_round=1500,
                evals=evals if evals else [],
                verbose_eval=200,
            )
            # Добавляем обученную модель в список
            trained_models_xgb.append(model)

            if Cfg.IS_VALIDATE:
                xgb_va_preds_1 = model.predict(dval)
                all_preds.append(xgb_va_preds_1)
                # # Evaluate XGBoost
                xgb_hr3 = self.hitrate_at_3(y_va, xgb_va_preds_1, groups_va)
                print(f"HitRate@3: {xgb_hr3:.8f}")
                
                ensemble_preds = np.sum(all_preds, axis=0)
                xgb_hr3 = self.hitrate_at_3(y_va, ensemble_preds, groups_va)
                print(f"HitRate@3 All: {xgb_hr3:.8f}")
            
            # Возвращаем num_boost_round обратно в params для возможного дальнейшего использования
            params['num_boost_round'] = num_boost_round

        print(f"\nTotal models trained: {len(trained_models_xgb)}")
        print("Сохраняем модели")
        # Сохранение всего списка models
        with open(Cfg.models_path, "wb") as f:
            pickle.dump(trained_models_xgb, f)
        

    def predict(self, test_path):
        '''
        Делает предсказание.
        test_path путь к датафрейму с данными, по которым нужно сделать предсказание
        '''

        test = pl.read_parquet(test_path).drop('__index_level_0__').with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))
        dtest, _, _, _ = get_dmatrix(test)

        with open(Cfg.models_path, "rb") as f:
            trained_models_xgb = pickle.load(f)


        all_preds = []  # Список для хранения предсказаний каждой модели
        
        for model in trained_models_xgb:
            preds = model.predict(dtest)
            all_preds.append(preds)

        # Суммируем все предсказания
        ensemble_preds = np.sum(all_preds, axis=0)
        
        return ensemble_preds