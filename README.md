Hello!

Below you can find a outline of how to reproduce my solution for the aeroclub-recsys-2025 competition.
If you run into any trouble with the setup/code or have any questions please contact me at mikhail.golubchik@gmail.com

# ARCHIVE CONTENTS
- data                     : contains data to train and test model
- models                   : contains trained models
- submissions              : contains predictopn on test
- common.py                : code of config class
- prepare_data.py          : code of prepairing data
- model.py                 : code of the model class
- train.py                 : code for training models
- predict.py               : code for predicion on test

# HARDWARE:
(The following specs were used to create the original solution)  
Ubuntu 24.04.2 LTS (2 TB boot disk)  
AMD EPYC 7452 32-Core Processor ( 64 vCPUs, 128 GB memory)  
4 x NVIDIA RTX A5000 (not used in this compettion)  

# SOFTWARE
(python packages are detailed separately in `requirements.txt`):  
Python 3.11.11  

# DATA SETUP
(assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)  
below are the shell commands used in each step, as run from the top level directory  
mkdir -p data  
cd data  
kaggle competitions download -c aeroclub-recsys-2025 -f train.parquet  
kaggle competitions download -c aeroclub-recsys-2025 -f test.parquet  


# DATA PROCESSING
python python prepare_data.py

# MODEL BUILD:
python train.py

# MAKE PREDICTIONS:
python predict.py