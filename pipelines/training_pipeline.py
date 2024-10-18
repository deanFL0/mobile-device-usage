from zenml import pipeline
from steps.load_data import load_data
from steps.preprocess_data import preprocess_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

@pipeline(enable_cache=True)
def training_pipeline(data_path: str):
    """
    Training pipeline
    """
    data = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)