import os

from src.model.architecture import build_model
from src.data.loader import download_data, load_data
from src.data.preprocess import preprocess_data

URL = 'https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv'
DEST = 'data/raw'
INSURANCE_CSV = 'insurance.csv'
CATEGORICAL_COLUMNS = ['sex', 'smoker', 'region']
NUMERIC_COLUMNS = ['age', 'bmi', 'children']

def train_model(data_url, dest_dir, csv_file, cat_col, num_col):
    download_data(data_url, dest_dir)
    file_path = os.path.join(dest_dir, csv_file)

    dataset = load_data(file_path)
    train_dataset, train_labels = preprocess_data(cat_col, num_col, dataset)

    input_dim = train_dataset.shape[1]
    model = build_model(input_dim)
    model.fit(train_dataset, train_labels, epochs=100, batch_size=32, shuffle=True)

    os.makedirs('models', exist_ok=True)
    model_path = 'models/health-costs-calculator.h5'
    model.save(model_path)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model(URL, DEST, INSURANCE_CSV, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS)

