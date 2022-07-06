from functional import print_score, roc_auc_score, f1_score, print_dict
from .base import BasePipepine

import pandas as pd
import pickle as pkl


class TFPipeline(BasePipepine):
    def __init__(self, model, model_name: str):
        super().__init__()
        self.model = model
        self.model_name = model_name

    def train(self, x, y, epochs, batch_size, verbose=1):
        self.model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size, verbose=verbose)

        losses = pd.DataFrame(self.model.history.history)

        history_save_path = f"checkpoints/{self.model_name}/history.pkl"
        with open(history_save_path, "wb") as f:
            pkl.dump(losses, f)
    
    def load_model(self):
        model_save_dir = f"checkpoints/{self.model_name}/model"
        raise NotImplementedError()

    def save_model(self):
        model_save_dir = f"checkpoints/{self.model_name}/model"
        self.model.save(model_save_dir)
        print("Save model at:", model_save_dir)

    def predict(self, **kwargs):
        return self.model(**kwargs)

    def get_history(self):
        return self.model.history.history

    def test(self, X_test, y_test):
        predictions = (self.model.predict(X_test) > 0.5).astype("int32")

        info = {
            "test_score" : print_score(y_test, predictions, train=False),
            "f1" : f1_score(y_test, predictions),
            "roc_auc_score" : roc_auc_score(predictions, y_test)
        }

        print_dict(info)

        info_save_path = f"checkpoints/{self.model_name}/info.pkl"
        with open(info_save_path, 'wb') as f:
            pkl.dump(info, f)
        print("Save information at:", info_save_path)