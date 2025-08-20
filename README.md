Hello!

Below you can find a outline of how to reproduce my solution for the **aeroclub-recsys-2025 competition**.<br>
If you run into any trouble with the setup/code or have any questions please contact me at mikhail.golubchik@gmail.com<br>

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
(The following specs were used to create the original solution)<br>
- Ubuntu 24.04.2 LTS (2 TB boot disk)  
- AMD EPYC 7452 32-Core Processor ( 64 vCPUs, 128 GB memory)  
- 4 x NVIDIA RTX A5000 (not used in this compettion)  

# SOFTWARE
(python packages are detailed separately in `requirements.txt`):<br>
Python 3.11.11<br>

# DATA SETUP
(assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)<br>
below are the shell commands used in each step, as run from the top level directory  

mkdir -p data<br>
cd data<br>
kaggle competitions download -c aeroclub-recsys-2025 -f train.parquet<br>
kaggle competitions download -c aeroclub-recsys-2025 -f test.parquet<br>


# DATA PROCESSING
python python prepare_data.py

# MODEL BUILD:
python train.py

# MAKE PREDICTIONS:
python predict.py