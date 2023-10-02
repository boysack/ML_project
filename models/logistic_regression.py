from abc import abstractmethod
import numpy as np
import scipy.optimize as opt

class discriminative_model:
    def __init__(self, data:np.ndarray, labels:np.ndarray, l:int) -> None:
        self.data = data
        self.labels = labels
        self.l = l
        self.weights = None
        self.bias = None
        self.score_values = None
        self.is_fitted = False

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def scores():
        pass

class logistic_regression(discriminative_model):
    def __init__(self, data:np.ndarray, labels:np.ndarray, l:int) -> None:
        super().__init__(data, labels, l)

    def train(self):
        J = self.logreg_obj_wrap()
        x0 = np.zeros(self.data.shape[0] + 1)
        result_of_minimizer, _, _ = opt.fmin_l_bfgs_b(J, x0, approx_grad=True)
        # print("result of minimizer: " + str(result_of_minimizer))
        self.w = result_of_minimizer[:-1]
        self.b = result_of_minimizer[-1]
        # print("w: " + str(self.w))
        # print("b: " + str(self.b))
        self.is_fitted = True

    def scores(self):
        if(self.is_fitted is False):
            self.train()
        self.score_values = np.dot(self.w.T, self.data) + self.b
        # print("score_values_size : " + str(self.score_values.shape))

    def logreg_obj_wrap(self):

        def logreg_obj(params:np.ndarray):
            # print("params: " + str(params))
            w, b = params[:-1], params[-1]
            z = 2 * self.labels - 1
            regularization_term = (self.l / 2) * np.dot(w, w.T)
            inner_sum = 0

            for i in range(self.data.shape[1]):
                # print("term: " + str((np.dot(w.T, self.data[:, i]) + b)))
                # print("size: " + str((np.dot(w.T, self.data[:, i]) + b).shape)) # should be an array of size 1
                second_term_of_log = -z[i] * (np.dot(w.T, self.data[:, i]) + b)
                inner_sum += np.logaddexp(0, second_term_of_log)
            
            value_to_return = regularization_term + inner_sum / self.data.shape[1]
            return value_to_return
        return logreg_obj
