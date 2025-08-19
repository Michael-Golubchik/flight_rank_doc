# AeroClub RecSys 2025 - XGBoost Ranking

# This notebook implements an improved ranking approach using XGBoost and Polars for the AeroClub recommendation challenge.

# Based on AeroClub RecSys 2025 - XGBoost Ranking Baseline from Kirill Khoruzhii
# https://www.kaggle.com/code/ka1242/xgboost-ranker-with-polars


import polars as pl
import numpy as np

# import time
import xgboost as xgb
from pathlib import Path

import random
import pickle

from datetime import datetime
from tqdm import tqdm

from common import Cfg


## Feature Engineering
### Обработка категориальных переменных
# Категориальные признаки
cat_features_final = [
    'nationality',
    'companyID',
    'corporateTariffCode',
    # 'bySelf',
    # 'sex',
    'legs0_segments0_aircraft_code',
    'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_arrivalTo_airport_iata',
    'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_marketingCarrier_code',
    'legs0_segments0_operatingCarrier_code',
    'legs0_segments1_aircraft_code',
    'legs0_segments1_arrivalTo_airport_city_iata',
    'legs0_segments1_arrivalTo_airport_iata',
    'legs0_segments1_departureFrom_airport_iata',
    'legs0_segments1_marketingCarrier_code',
    'legs0_segments1_operatingCarrier_code',
    'legs0_segments2_aircraft_code',
    'legs0_segments2_arrivalTo_airport_city_iata',
    'legs0_segments2_arrivalTo_airport_iata',
    'legs0_segments2_departureFrom_airport_iata',
    'legs0_segments2_marketingCarrier_code',
    'legs0_segments2_operatingCarrier_code',
    'legs1_segments0_aircraft_code',
    'legs1_segments0_arrivalTo_airport_city_iata',
    'legs1_segments0_arrivalTo_airport_iata',
    'legs1_segments0_departureFrom_airport_iata',
    'legs1_segments0_marketingCarrier_code',
    'legs1_segments0_operatingCarrier_code',
    'legs1_segments1_aircraft_code',
    'legs1_segments1_arrivalTo_airport_city_iata',
    'legs1_segments1_arrivalTo_airport_iata',
    'legs1_segments1_departureFrom_airport_iata',
    'legs1_segments1_marketingCarrier_code',
    'legs1_segments1_operatingCarrier_code',
    'legs1_segments2_aircraft_code',
    'legs1_segments2_arrivalTo_airport_city_iata',
    'legs1_segments2_arrivalTo_airport_iata',
    'legs1_segments2_departureFrom_airport_iata',
    'legs1_segments2_marketingCarrier_code',
    'legs1_segments2_operatingCarrier_code',
    'legs0_segments0_flightNumber',
    'legs0_segments1_flightNumber',
    'legs0_segments2_flightNumber',
    'legs1_segments0_flightNumber',
    'legs1_segments1_flightNumber',
    'legs1_segments2_flightNumber',
]


def encode_categories(df: pl.DataFrame, cat_cols: list[str]) -> tuple[pl.DataFrame, dict[str, dict[str, int]]]:
    '''
    Создаёт маппинг для категориальных колонок и кодирует их в Int16,
    заменяя неизвестные на -1. Значение "missing" всегда кодируется как -1.
    '''
    cat_map = {}
    for col in cat_cols:
        unique_vals = df[col].drop_nulls().unique().to_list()
        
        # Определяем тип колонки
        col_dtype = df.schema[col]

        # Создаем маппинг в зависимости от типа данных
        if col_dtype == pl.Utf8:
            # Для строковых колонок: "missing" → -1
            unique_vals = [v for v in unique_vals if v != "missing"]
            mapping = {"missing": -1}
        else:
            # Для числовых колонок: -1 → -1
            unique_vals = [v for v in unique_vals if v != -1]
            mapping = {-1: -1}

        mapping.update({v: i for i, v in enumerate(unique_vals)})

        cat_map[col] = mapping
        max_index = -1  # используется как default для неизвестных

        df = df.with_columns([
            pl.col(col)
            .replace_strict(mapping, default=max_index)
            .cast(pl.Int16)
            .alias(col)
        ])

    with open(Cfg.cat_map_path, "wb") as f:
        pickle.dump(cat_map, f)

    return df


def apply_category_map(df: pl.DataFrame) -> pl.DataFrame:
    '''
    Применяет ранее созданный маппинг к другому DataFrame (например, test), подставляя -1 для unseen значений.
    '''
    with open(Cfg.cat_map_path, "rb") as f:
        cat_map= pickle.load(f)

    for col, mapping in cat_map.items():
        #max_index = max(mapping.values()) + 1
        max_index = -1
        df = df.with_columns([
            pl.col(col)
            .replace_strict(mapping, default=max_index)
            .cast(pl.Int16)
            .alias(col)
        ])

    return df


### Создание исторических признаков по клиентам
# More efficient duration to minutes converter
def dur_to_min(col):
    # Extract days and time parts in one pass
    days = col.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 1440
    time_str = pl.when(col.str.contains(r"^\d+\.")).then(col.str.replace(r"^\d+\.", "")).otherwise(col)
    hours = time_str.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
    minutes = time_str.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)
    return (days + hours + minutes).fill_null(0)


# def make_history_avg(df, source_col, target_col, group_col):
def make_history_avg(df, source_cols, group_col, suffix):
    '''
    Добавляет историческую информацию по выборам билетов для пользователя или компании
    для числовых колонок
    '''
    
    # Создаем датафрейм с выбранными записями (selected=1)
    selected_df = df.filter(pl.col("selected") == 1).select(["ranker_id", "requestDate", group_col] + source_cols)
    
    # Создаем словари для быстрого доступа
    ranker_to_profile = dict(zip(
        selected_df["ranker_id"].to_list(),
        selected_df[group_col].to_list()
    ))

    # Словарь ranker_id -> requestDate
    ranker_to_timestamp = dict(zip(
        selected_df["ranker_id"].to_list(),
        selected_df["requestDate"].to_list()
    ))
    
    # Создаем датафрейм с историческими данными (исключая текущие группы)
    history_df = selected_df.select(["ranker_id", "requestDate", group_col] + source_cols)
    
    all_stats_dict = {}  # ranker_id -> {col_mean: val, col_std: val}
    
    # Получаем уникальные значения ranker_id
    unique_ranker_ids = df["ranker_id"].unique().to_list()
        
    # Проходим по каждой группе
    for current_ranker_id in tqdm(unique_ranker_ids, desc="Обработка ranker_id", mininterval=10.0):
        # Получаем profile_id из словаря
        current_profile_id = ranker_to_profile.get(current_ranker_id)

        # timestamp текущей группы
        current_timestamp = ranker_to_timestamp.get(current_ranker_id)
        
        # Фильтруем исторические данные для текущего профиля
        profile_history = history_df.filter(
            (pl.col(group_col) == current_profile_id) &
            (pl.col("ranker_id") != current_ranker_id)
            & (pl.col("requestDate") < current_timestamp)
        )

        
        # Вычисляем статистики для всех колонок сразу
        agg_result = profile_history.select([
            *[pl.col(col).mean().alias(f"{col}{suffix}_mean") for col in source_cols],
            *[pl.col(col).std().alias(f"{col}{suffix}_std") for col in source_cols],
            *[pl.col(col).count().cast(pl.Int16).alias(f"{col}{suffix}_count") for col in source_cols],  # Добавляем счётчики
            *[pl.col(col).median().alias(f"{col}{suffix}_median") for col in source_cols], 
            *[pl.col(col).quantile(0.25).alias(f"{col}{suffix}_q25") for col in source_cols],
            *[pl.col(col).quantile(0.75).alias(f"{col}{suffix}_q75") for col in source_cols]
        ])

        if agg_result.height > 0:
            row = agg_result.row(0)
            n_cols = len(source_cols)
            all_stats_dict[current_ranker_id] = {
                **{
                    f"{col}{suffix}_mean": row[i] if row[i] is not None else -1
                    for i, col in enumerate(source_cols)
                },
                **{
                    f"{col}{suffix}_std": row[i+n_cols] if row[i+n_cols] is not None else -1
                    for i, col in enumerate(source_cols)
                },
                **{
                    f"{col}{suffix}_count": row[i+2*n_cols] if row[i+2*n_cols] is not None else -1  # Добавляем счётчики
                    for i, col in enumerate(source_cols)
                },
                **{
                    f"{col}{suffix}_median": row[i+3*n_cols] if row[i+3*n_cols] is not None else -1  # Добавляем счётчики
                    for i, col in enumerate(source_cols)
                },
                **{
                    f"{col}{suffix}_q25": row[i+4*n_cols] if row[i+4*n_cols] is not None else -1  # Добавляем счётчики
                    for i, col in enumerate(source_cols)
                },
                **{
                    f"{col}{suffix}_q75": row[i+5*n_cols] if row[i+5*n_cols] is not None else -1  # Добавляем счётчики
                    for i, col in enumerate(source_cols)
                },
            }
        else:
            all_stats_dict[current_ranker_id] = {
                **{f"{col}{suffix}_mean": -1 for col in source_cols},
                **{f"{col}{suffix}_std": -1 for col in source_cols},
                **{f"{col}{suffix}_count": -1 for col in source_cols}  # По умолчанию 0 записей
                **{f"{col}{suffix}_median": -1 for col in source_cols}  # По умолчанию 0 записей
                **{f"{col}{suffix}_q25": -1 for col in source_cols}  # По умолчанию 0 записей
                **{f"{col}{suffix}_q75": -1 for col in source_cols}  # По умолчанию 0 записей
            }
    
    # Создаем датафрейм для обновления
    update_data = []
    for ranker_id, stats in all_stats_dict.items():
        row = {"ranker_id": ranker_id, **stats}
        update_data.append(row)
    
    update_df = pl.DataFrame(update_data)
    
    # Обновляем основной датафрейм
    df = df.join(update_df, on="ranker_id", how="left")
    
    # Заполняем пропуски и приводим типы
    for col in source_cols:
        df = df.with_columns([
            pl.col(f"{col}{suffix}_mean").fill_null(-1),
            pl.col(f"{col}{suffix}_std").fill_null(-1),
            pl.col(f"{col}{suffix}_count").cast(pl.Int16).fill_null(-1),
            pl.col(f"{col}{suffix}_median").fill_null(-1),
            pl.col(f"{col}{suffix}_q25").fill_null(-1),
            pl.col(f"{col}{suffix}_q75").fill_null(-1),
        ])

    # Создаем список агрегаций
    agg_exprs = []

    for col in source_cols:
        # Среднее значение
        agg_exprs.append(pl.col(col).mean().alias(f"{col}{suffix}_mean"))
    for col in source_cols:
        # Стандартное отклонение
        agg_exprs.append(pl.col(col).std().alias(f"{col}{suffix}_std"))
    for col in source_cols:
        # Количество записей (не null)
        agg_exprs.append(pl.col(col).count().alias(f"{col}{suffix}_count"))
    for col in source_cols:
        # Количество записей (не null)
        agg_exprs.append(pl.col(col).median().alias(f"{col}{suffix}_median"))
    for col in source_cols:
        # Количество записей (не null)
        agg_exprs.append(pl.col(col).quantile(0.25).alias(f"{col}{suffix}_q25"))
    for col in source_cols:
        # Количество записей (не null)
        agg_exprs.append(pl.col(col).quantile(0.75).alias(f"{col}{suffix}_q75"))

    # Применяем агрегации
    df_stats = (
        df.filter(pl.col("selected") == 1)
        .group_by(group_col)
        .agg(agg_exprs)
    )
    # df_stats = to_32(df_stats)

    return df, df_stats


### Генерация новых признаков
def make_fetures(df, is_train=False):
    
    # Time features - batch process
    time_exprs = []
    for col in ("legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"):
        if col in df.columns:
            dt = pl.col(col).str.to_datetime(strict=False)
            h = dt.dt.hour().fill_null(12)
            time_exprs.extend([
                h.alias(f"{col}_hour"),
                dt.dt.weekday().fill_null(0).alias(f"{col}_weekday"),
                (((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20))).cast(pl.Int32).alias(f"{col}_business_time")
            ])
    if time_exprs:
        df = df.with_columns(time_exprs)

    # Process duration columns
    dur_cols = ["legs0_duration", "legs1_duration"] + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1, 2)]
    dur_exprs = [dur_to_min(pl.col(c)).alias(c) for c in dur_cols if c in df.columns]

    # Apply duration transformations first
    if dur_exprs:
        df = df.with_columns(dur_exprs)

    # Precompute marketing carrier columns check
    mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' for l in (0, 1) for s in range(4)]
    mc_exists = [col for col in mc_cols if col in df.columns]

    # Combine all initial transformations
    df = df.with_columns([
            # Price features
            (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
            (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
            pl.col("totalPrice").log1p().alias("log_price"),
            
            # Duration features
            (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
            pl.when(pl.col("legs1_duration").fill_null(0) > 0)
                .then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1))
                .otherwise(1.0).alias("duration_ratio"),
            
            # Trip type
            (pl.col("legs1_duration").is_null() | 
            (pl.col("legs1_duration") == 0) | 
            pl.col("legs1_segments0_departureFrom_airport_iata").is_null()).cast(pl.Int32).alias("is_one_way"),
            
            # Total segments count
            (pl.sum_horizontal(pl.col(col).is_not_null().cast(pl.UInt8) for col in mc_exists) 
            if mc_exists else pl.lit(0)).alias("l0_seg"),
            
            # FF features
            (pl.col("frequentFlyer").fill_null("").str.count_matches("/") + 
            (pl.col("frequentFlyer").fill_null("") != "").cast(pl.Int32)).alias("n_ff_programs"),
            
            # Binary features
            pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
            (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),
            
            # Baggage & fees
            (pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(0) + 
            pl.col("legs1_segments0_baggageAllowance_quantity").fill_null(0)).alias("baggage_total"),

            (
                (pl.col("miniRules0_monetaryAmount") == 0)
                & (pl.col("miniRules0_statusInfos") == 1)
            )
            .cast(pl.Int8)
            .alias("free_cancel"),
            (
                (pl.col("miniRules1_monetaryAmount") == 0)
                & (pl.col("miniRules1_statusInfos") == 1)
            )
            .cast(pl.Int8)
            .alias("free_exchange"),
            
            # Routes & carriers
            pl.col("searchRoute").is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"])
                .cast(pl.Int32).alias("is_popular_route"),
            
            # Cabin
            pl.mean_horizontal(["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]).alias("avg_cabin_class"),
            (pl.col("legs0_segments0_cabinClass").fill_null(0) - 
            pl.col("legs1_segments0_cabinClass").fill_null(0)).alias("cabin_class_diff"),
    ])

    # Segment counts - more efficient
    seg_exprs = []
    for leg in (0, 1):
        seg_cols = [f"legs{leg}_segments{s}_duration" for s in range(4) if f"legs{leg}_segments{s}_duration" in df.columns]
        if seg_cols:
            seg_exprs.append(
                pl.sum_horizontal(pl.col(c).is_not_null() for c in seg_cols)
                    .cast(pl.Int32).alias(f"n_segments_leg{leg}")
            )
        else:
            seg_exprs.append(pl.lit(0).cast(pl.Int32).alias(f"n_segments_leg{leg}"))

    # Add segment-based features
    # First create segment counts
    df = df.with_columns(seg_exprs)

    # Then use them for derived features
    df = df.with_columns([
        (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias("total_segments"),
        (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
        pl.when(pl.col("is_one_way") == 1).then(0)
            .otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32)).alias("is_direct_leg1"),
    ])

    # More derived features
    df = df.with_columns([
        (pl.col("is_direct_leg0") & pl.col("is_direct_leg1")).cast(pl.Int32).alias("both_direct"),
        ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0)).cast(pl.Int32).alias("is_vip_freq"),
        (pl.col("baggage_total") > 0).cast(pl.Int32).alias("has_baggage"),
        pl.col("Id").count().over("ranker_id").alias("group_size"),
    ])

    # Add major carrier flag if column exists
    if "legs0_segments0_marketingCarrier_code" in df.columns:
        df = df.with_columns(
            pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7", "U6"])
                .cast(pl.Int32).alias("is_major_carrier")
        )
    else:
        df = df.with_columns(pl.lit(0).alias("is_major_carrier"))

    df = df.with_columns(pl.col("group_size").log1p().alias("group_size_log"))

    # Batch rank computations - more efficient with single pass
    # First apply the columns that will be used for ranking
    df = df.with_columns([
        pl.col("group_size").log1p().alias("group_size_log"),
    ])

    # Price and duration basic ranks
    rank_exprs = []
    for col, alias in [("totalPrice", "price"), ("total_duration", "duration")]:
        rank_exprs.append(pl.col(col).rank().over("ranker_id").alias(f"{alias}_rank"))

    # Price-specific features
    price_exprs = [
        (pl.col("totalPrice").rank("average").over("ranker_id") / 
        pl.col("totalPrice").count().over("ranker_id")).alias("price_pct_rank"),
        (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")).cast(pl.Int32).alias("is_cheapest"),
        ((pl.col("totalPrice") - pl.col("totalPrice").median().over("ranker_id")) / 
        (pl.col("totalPrice").std().over("ranker_id") + 1)).alias("price_from_median"),
        ((pl.col("total_duration") - pl.col("total_duration").median().over("ranker_id")) / 
        (pl.col("total_duration").std().over("ranker_id") + 1)).alias("duration_from_median"),
        (pl.col("l0_seg") == pl.col("l0_seg").min().over("ranker_id")).cast(pl.Int32).alias("is_min_segments"),
    ]

    # Apply initial ranks
    df = df.with_columns(rank_exprs + price_exprs)

    # Cheapest direct - more efficient
    direct_cheapest = (
        df.filter(pl.col("is_direct_leg0") == 1)
        .group_by("ranker_id")
        .agg(pl.col("totalPrice").min().alias("min_direct"))
    )

    df = df.join(direct_cheapest, on="ranker_id", how="left").with_columns(
        ((pl.col("is_direct_leg0") == 1) & 
        (pl.col("totalPrice") == pl.col("min_direct"))).cast(pl.Int32).fill_null(0).alias("is_direct_cheapest")
    ).drop("min_direct")

    # Popularity features - efficient join
    df = (
        df.join(
            df.group_by('legs0_segments0_marketingCarrier_code').agg(pl.mean('selected').alias('carrier0_pop')),
            on='legs0_segments0_marketingCarrier_code', 
            how='left'
        )
        .join(
            df.group_by('legs1_segments0_marketingCarrier_code').agg(pl.mean('selected').alias('carrier1_pop')),
            on='legs1_segments0_marketingCarrier_code', 
            how='left'
        )
        .with_columns([
            pl.col('carrier0_pop').fill_null(0.0),
            pl.col('carrier1_pop').fill_null(0.0),
        ])
    )

    # Final features including popularity
    df = df.with_columns([
        (pl.col('carrier0_pop') * pl.col('carrier1_pop')).alias('carrier_pop_product'),
    ])

    source_cols=['legs0_departureAt_hour','legs1_departureAt_hour','legs0_arrivalAt_hour','legs1_arrivalAt_hour',
                 'price_rank', 'price_from_median', 'duration_rank', 'avg_cabin_class',
                 'baggage_total', 'l0_seg',
                 'legs0_segments0_seatsAvailable', 'legs0_segments0_baggageAllowance_quantity', 'legs0_segments0_cabinClass',
                 'miniRules1_statusInfos', 'miniRules0_statusInfos',
                 'duration_from_median',
                 ]

    if is_train:
        print("avg на начало:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        df, df_stats_pr = make_history_avg(df, source_cols=source_cols, group_col="profileId", suffix='_pr')
        df, df_stats_co = make_history_avg(df, source_cols=source_cols, group_col="companyID", suffix='_co')
        print("avg на завершение:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Считаем количество уникальных ranker_id для каждого profileId
        # и сохраняем в profile_counts для использования в валидационном или тестовом датасете
        profile_counts = (
            df
            .group_by("profileId")
            .agg(pl.col("ranker_id").n_unique().alias("ranker_count"))
        )

        # Делаем счетчики с нуля не используя
        df = df.with_columns(
            pl.col("ranker_id")
            .n_unique()
            .over("profileId")
            .cast(pl.Int16)
            .alias("ranker_count")
        )

        # сохраняем статистики
        df_stats_pr.write_parquet(Path(Cfg.stats_dir) / "df_stats_pr.parquet", compression="zstd")
        df_stats_co.write_parquet(Path(Cfg.stats_dir) / "df_stats_co.parquet", compression="zstd")
        profile_counts.write_parquet(Path(Cfg.stats_dir) / "profile_counts.parquet", compression="zstd")

    else:
        # загружаем статистики
        df_stats_pr = pl.read_parquet(Path(Cfg.stats_dir) / "df_stats_pr.parquet")
        df_stats_co = pl.read_parquet(Path(Cfg.stats_dir) / "df_stats_co.parquet")
        profile_counts = pl.read_parquet(Path(Cfg.stats_dir) / "profile_counts.parquet")

        # Присоединяем статистики фичей по отдельным клиентам
        df = df.join(
            df_stats_pr,
            on="profileId",
            how="left"
        )

        # Присоединяем статистики фичей по компаниям клиентов
        df = df.join(
            df_stats_co,
            on="companyID",
            how="left"
        )

        # Присоединяем profile_counts к new_df
        df = df.join(profile_counts, on="profileId", how="left")

        # Заполняем пропуски 1
        df = df.with_columns(
            pl.col("ranker_count").fill_null(1)
        )

    print ('Обрабатываем пропуски')
    
    for col in df.select(pl.selectors.numeric()).columns:
        df = df.with_columns(pl.col(col).fill_null(-1).alias(col))

    for col in df.select(pl.selectors.string()).columns:
        df = df.with_columns(pl.col(col).fill_null("missing").alias(col))

    return df


## Feature Selection
def select_fetures(data):
    # Возвращает data еще со всеми столбцами
    # В feature_cols указаны те столбцы что нужно потом оставить для обучения

    # Columns to exclude (uninformative or problematic)
    exclude_cols = [
        'Id', 'ranker_id', 'selected', 'profileId',     'requestDate',
        'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
        'miniRules0_percentage', 'miniRules1_percentage',  # >90% missing
        'frequentFlyer',  # Already processed
        'group_size',
        'searchRoute',
        # Exclude constant columns
        'pricingInfo_passengerCount', 
    ]

    # Exclude segment 3 columns (>98% missing)
    for leg in [0, 1]:
        for seg in [3]:
            for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                        'baggageAllowance_quantity', 'baggageAllowance_weightMeasurementType',
                        'cabinClass', 'departureFrom_airport_iata', 'duration', 'flightNumber',
                        'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
                exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')

    # # Исключаем слишком уникальные фичи
    # for leg in [0, 1]:
    #     for seg in [0, 1, 2]:
    #         for suffix in ['flightNumber',
    #                        #'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata', 'departureFrom_airport_iata'
    #                        ]:
    #             exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')

    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features ({len(cat_features_final)} categorical)")
    groups = data.select('ranker_id')

    # return data, groups, feature_cols
    return data, groups, feature_cols#, cat_features_final


def get_dmatrix(df):
    '''
    Генерирует DMatrix для XGBoost
    '''
    df = make_fetures(df)
    
    df, groups_df, feature_cols = select_fetures(df)

    y_df = df.select('selected')
    df = df.select(feature_cols)
    # Применяем маппинг к df
    df = apply_category_map(df)
    group_sizes_df = groups_df.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()
    feature_names=list(df.columns)
    
    d_df = xgb.DMatrix(df, label=y_df, missing=-1, group=group_sizes_df, feature_names=feature_names)

    return d_df, y_df, groups_df, df


if __name__ == "__main__":
    print('_________________________________________________________________________________________________________\n')

    ## Подготовка к тренировке моделей
    ### Prepair Data
    # Load data
    train = pl.read_parquet(Cfg.train_path).drop('__index_level_0__')

    if Cfg.IS_VALIDATE:
        n2 = train.height
        n1 = Cfg.val_start_id if Cfg.IS_VALIDATE else train.height
        validate=train[n1:n2]
        train = train[:n1]

    requestDate_tr = train.select('requestDate')
    profileId_tr =  train.select('profileId')


    print('make new features')
    # Создаем новые признаки
    train = make_fetures(train, is_train=True)

    # Выбираем необходимые для дальнейшего обучения признаки
    train, groups_tr, feature_cols = select_fetures(train)

    print('feature_cols:', feature_cols)

    y_tr = train.select('selected')
    train = train.select(feature_cols)

    # Кодируем train
    train = encode_categories(train, cat_features_final)

    # Исходные данные до удаления столбцов и строк для тренировки
    train_source = train.select(feature_cols).with_columns(
        groups_tr["ranker_id"],
        y_tr["selected"],
        requestDate_tr["requestDate"],
        profileId_tr["profileId"]
    )

    print('Save train')
    # Сохраняем train
    train_source.write_parquet(Cfg.train_path_processed, compression="zstd")

    if Cfg.IS_VALIDATE:
        validate = validate.filter(
            pl.len().over("ranker_id") > 10
        )
        dval, y_va, groups_va, validate = get_dmatrix(validate)

        # Исходные данные до удаления столбцов и строк для валидации
        val_source = validate.with_columns(
            groups_va["ranker_id"],
            y_va["selected"],
        )
        print('Save validate')
        val_source.write_parquet(Cfg.valid_path_processed, compression="zstd")
