import numpy as np

class CrossEntropyLoss:
    def forward(self, preds, actuals):
        assert preds.size == actuals.size and preds.shape == actuals.shape

        # print("Loss: ", -1. * np.sum(actuals * np.log(preds)))
        return -1. * np.mean(actuals * np.log(preds))
    
    def backward(self, preds, actuals):
        # print("Loss gradients", ((preds - actuals)).tolist())
        return (preds - actuals)


class MSE:
    def forward(cls, preds, actuals):
        assert preds.size == actuals.size and preds.shape == actuals.shape

        return ((actuals - preds) ** 2).mean()
    
    def backward(cls, preds, actuals):

        return -2 * (actuals - preds)