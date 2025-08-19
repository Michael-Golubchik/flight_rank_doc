# Entry Points

1. `python prepare_data.py`
  - Read training data from RAW_DATA_DIR (specified in SETTINGS.json)
  - Run preprocessing steps
  - Save the prepaired data to PROCESSED_DATA_DIR (specified in SETTINGS.json)
  - Save the statistics to STATS_DIR (specified in SETTINGS.json)

2. `python train.py`
  - Read training data from PROCESSED_DATA_DIR (specified in SETTINGS.json)
  - Train model.
  - Save model to MODEL_DIR (specified in SETTINGS.json)
  
3. `python predict.py`
  - Read test data from RAW_DATA_DIR (specified in SETTINGS.json)
  - Load your model from MODEL_DIR (specified in SETTINGS.json)
  - Use model to make predictions on new samples
  - Save predictions to SUBMISSION_DIR (specified in SETTINGS.json)
