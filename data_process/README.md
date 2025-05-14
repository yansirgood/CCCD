# Data Process and Data Generate 
Data processing is divided into training data processing and other data processing

Note: There needs to be a certain order when cleaning data. The training set needs to be cleaned first, and then other data.
## Train Data Process
* Prepare the train data files,
* Directly run the `./data_rocess/train_data_process.py`. Here is an example :
    ```bash
    cd data_rocess
    Train_Data='xxx'
    Python3 train_data_process.py
    ```
  * `--Train_Data`: The path for Train data
## Data Generate
* Prepare the train data files,
* Directly run the `./data_rocess/data_generate.py`. Here is an example :
    ```bash
    cd data_rocess
    Config_path='xxx'
    Python3 data_generate.py
    Python3 txt_to_xlsx
    Python3 xlsx_to_json
    ```
  * `--config_path`: The path for config of model's api
## Other Data Process
* Prepare the train data files,
* Directly run the `./data_rocess/other_data_process.py`. Here is an example :
   ```bash
    cd data_rocess
    Other_Data='xxx'
    Python3 other_data_process.py
    ```
    * `--Other_Data`: The path for Other data



