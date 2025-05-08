from src.model.evaluate import evaluate_model

MODEL_PATH = 'models/health-costs-calculator.h5'
INSURANCE_DIR = 'data/raw/insurance.csv'
CATEGORICAL_COLUMNS = ['sex', 'smoker', 'region']
NUMERIC_COLUMNS = ['age', 'bmi', 'children']

def test_health_costs_model():
    mae = evaluate_model(MODEL_PATH, INSURANCE_DIR, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS)

    assert mae < 3500, (
        f"Mean Abs Error = {mae:.2f}, debe ser menor que 3500. "
        "Keep trying."
    )
    