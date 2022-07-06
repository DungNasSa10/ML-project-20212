import pickle as pkl

from functional import print_score, roc_auc_score, f1_score, print_dict
from .base import BasePipepine

class SKPipeline(BasePipepine):
    def __init__(self, model, model_name: str) -> None:
        super().__init__()

        self.model = model
        self.model_name = model_name

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def load_model(self):
        model_save_path = f"checkpoints/{self.model_name}/model.pkl"
        with open(model_save_path, 'rb') as f:
            self.model = pkl.load(f)
        print("Save load at:", model_save_path)

    def save_model(self):
        model_save_path = f"checkpoints/{self.model_name}/model.pkl"
        with open(model_save_path, 'wb') as f:
            pkl.dump(self.model, f)
        print("Save model at:", model_save_path)

    def predict(self, **kwargs):
        pass

    def test(self, X_train, X_test, y_train, y_test):
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        info = {
            "train_score" : print_score(y_train, y_train_pred, train=True),
            "test_score" : print_score(y_test, y_test_pred, train=False),
            "f1" : f1_score(y_test, y_test_pred),
            "roc_auc_score" : roc_auc_score(y_test_pred, y_test),
        }

        print_dict(info)

        info_save_path = f"checkpoints/{self.model_name}/info.pkl"
        with open(info_save_path, 'wb') as f:
            pkl.dump(info, f)
        print("Save information at:", info_save_path)
