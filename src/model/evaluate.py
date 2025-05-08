from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

from src.data.loader import load_data
from src.data.preprocess import preprocess_data

MODEL_PATH = 'models/health-costs-calculator.h5'
INSURANCE_DIR = 'data/raw/insurance.csv'
CATEGORICAL_COLUMNS = ['sex', 'smoker', 'region']
NUMERIC_COLUMNS = ['age', 'bmi', 'children']

def evaluate_model(model_path, data_path, cat_col, num_col):
    dataset = load_data(data_path)
    test_dataset, test_labels = preprocess_data(cat_col, num_col, dataset, train=False)

    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})

    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
    print(f"Model Evaluation:\nLoss: {loss}\nMAE: {mae}\nMSE: {mse}")

    return mae

if __name__ == "__main__":
    evaluate_model(MODEL_PATH, INSURANCE_DIR, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS)


