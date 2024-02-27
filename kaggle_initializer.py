import os
import json

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(kaggle_username: str, kaggle_key: str, force_download: bool = False) -> None :
    """ Download the kaggle dataset to the data folder via access data.

    Args:
        kaggle_username (str): Access username.
        kaggle_key (str): Key to access kaggle.
        force_download (bool, optional): Wether we want to redownlaod the dataset if it already exists. Defaults to False.
    """
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key
    
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    
    train_and_test_exist = Path("/data/train").exists() and Path("/data/test").exists() 
    if not train_and_test_exist or force_download: kaggle_api.dataset_download_files(
        'samuelcortinhas/muffin-vs-chihuahua-image-classification', 'data', unzip=True
    )
    
    print('We have read the dataset correctly from Kaggle.\n'
          'The dataset was already unpacked in the data and test sets.')


def download_dataset_with_kagglejson(force_download: bool = False) -> None:
    kaggle_configuration_data = json.load(open('data/kaggle.json'))
    download_dataset(kaggle_configuration_data['username'], kaggle_configuration_data['key'], force_download)
   
download_dataset_with_kagglejson(False)