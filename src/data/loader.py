import os
import requests
import pandas as pd


def download_data(url, dest):
    if not os.path.exists(dest):
        os.makedirs(dest)

    response = requests.get(url)
    filename = os.path.basename(url)
    file_path = os.path.join(dest, filename)

    with open(file_path, 'wb') as f:
        f.write(response.content)

def load_data(file_path):
    return pd.read_csv(file_path)