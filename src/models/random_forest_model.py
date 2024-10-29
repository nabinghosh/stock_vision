from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def preprocess_data(self, data):
        X = data.drop('target', axis=1)
        y = data['target']
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    def train(self, X, y):
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, 
                                           max_depth=self.max_depth, 
                                           random_state=self.random_state)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        self.model = joblib.load(filepath)