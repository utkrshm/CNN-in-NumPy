import numpy as np

class CrossEntropyLoss:
    def forward(cls, preds, actuals):
        # print(cls, preds, actuals)
        assert preds.size == actuals.size and preds.shape == actuals.shape
        
        return -1. * np.mean(actuals * np.log(preds))
    
    def backward(cls, preds, actuals):
        # print(cls, preds, actuals)
        return -1. * (actuals / preds)


class MSE:
    def forward(cls, preds, actuals):
        assert preds.size == actuals.size and preds.shape == actuals.shape

        return ((actuals - preds) ** 2).mean()
    
    def backward(cls, preds, actuals):
        print(cls, preds, actuals)

        return -2 * (actuals - preds)