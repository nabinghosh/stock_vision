import xgboost as xgb
from sklearn.model_selection import train_test_split
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=100, random_state=42):
        super().__init__("XGBoost")
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state

    def preprocess_data(self, data):
        X = data.drop('target', axis=1)
        y = data['target']
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    def train(self, X, y):
        self.model = xgb.XGBRegressor(max_depth=self.max_depth,
                                      learning_rate=self.learning_rate,
                                      n_estimators=self.n_estimators,
                                      random_state=self.random_state)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filepath):
        self.model.save_model(filepath)

    def load_model(self, filepath):
        self.model = xgb.XGBRegressor()
        self.model.load_model(filepath)