import numpy as np

class CrossEntropyLoss:
    def forward(cls, preds, actuals):
        assert preds.size == actuals.size and preds.shape == actuals.shape
        
        return -1. * np.mean(actuals * np.log(preds))
    
    def backward(cls, preds, actuals):
        return (preds - actuals) / (preds * (1 - preds))


class MSE:
    def forward(cls, preds, actuals):
        assert preds.size == actuals.size and preds.shape == actuals.shape

        return ((actuals - preds) ** 2).mean()
    
    def backward(cls, preds, actuals):

        return -2 * (actuals - preds)