# Entry Points

- **Data**
  - `load_data.py` — загрузка сырых данных
  - `preprocess.py` — очистка и подготовка признаков
- **Training**
  - `train_xgb.py` — обучение XGBoost модели
  - `train_lgbm.py` — обучение LightGBM модели
- **Evaluation**
  - `cv.py` — кросс-валидация
  - `evaluate.py` — финальная оценка на тесте
- **Submission**
  - `predict.py` — предсказания на тесте
  - `make_submission.py` — формирование файла для LB
