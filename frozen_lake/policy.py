#
# differentiable policy class --> something to take actions in 2D
#
import numpy as np

class SoftmaxPolicy:
    def __init__(self):
        pass

    def softmax(self, logits):
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, probabilities, Y):
        n_samples = Y.shape[0]

        log_likelihood = -np.log(probabilities[range(n_samples), np.argmax(Y, axis=1)])
        return np.sum(log_likelihood, axis=1) / n_samples
    
    def derivative_single_sample(self, logits):
        p = self.softmax(logits)
        return np.diag(p) - np.outer(p, p)