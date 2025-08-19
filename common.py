import json
from pathlib import Path
import numpy as np

class Cfg:
    # Флаги валидации и загрузки
    IS_VALIDATE = True # Делать ли валидацию
    val_start_id = 14501077 # Индекс старта валидации в данных
    
    # Параметры модели
    #num_boost_round = 1200
    num_boost_round = 100
    RANDOM_STATE = 42

    
    # Инициализация случайных seed
    np.random.seed(RANDOM_STATE)
    
    @staticmethod
    def load_settings():
        # Загрузка SETTINGS.json
        with open("SETTINGS.json", "r") as f:
            settings = json.load(f)
        
        # Пути к данным
        Cfg.raw_data_dir = Path(settings["RAW_DATA_DIR"])
        Cfg.train_path = Path(settings["TRAIN_DATA_PATH"])
        Cfg.test_path = Path(settings["TEST_DATA_PATH"])
        Cfg.stats_dir = Path(settings["STATS_DIR"])
        Cfg.cat_map_path = Path(settings["CAT_MAP_PATH"])
        Cfg.train_path_processed = Path(settings["TRAIN_DATA_PROCESSED_PATH"])
        Cfg.valid_path_processed = Path(settings["VALID_DATA_PROCESSED_PATH"])
        
        # Пути для моделей и результатов
        Cfg.models_dir = Path(settings["MODEL_CHECKPOINT_DIR"])
        Cfg.models_path = Path(settings["MODEL_CHECKPOINT_PATH"])
        Cfg.logs_dir = Path(settings["LOGS_DIR"])
        Cfg.submissions_dir = Path(settings["SUBMISSION_DIR"])
        Cfg.submissions_path = Path(settings["SUBMISSION_PATH"])

# Загружаем настройки при импорте
Cfg.load_settings()

# убедимся, что папка существует
Path(Cfg.stats_dir).mkdir(parents=True, exist_ok=True)