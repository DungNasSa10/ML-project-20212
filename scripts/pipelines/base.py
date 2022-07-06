class BasePipepine:
    def preprocess(self, **kwargs):
        raise NotImplementedError()

    def train(self, **kwargs):
        raise NotImplementedError()

    def predict(self, **kwargs):
        raise NotImplementedError()
    
    def evaluate(self, **kwargs):
        raise NotImplementedError()
    
    def test(self, **args):
        raise NotImplementedError()