import evaluation
import numpy as np
import unittest


class TestEvaluationMethods(unittest.TestCase):

    def test_precision_at_n(self):
        real = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        yhat = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        exp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0/6.0, 4.0/7.0, 6.0/8.0, 8.0/9.0, 1.0])

        for i in range(len(real)):
            result = evaluation.precision_at_n(real, yhat, i+1)
            assert exp[i] == result
