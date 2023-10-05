from abc import abstractmethod
import numpy as np
import scipy.optimize as opt

class discriminative_model:
    def __init__(self, data:np.ndarray, labels:np.ndarray, data_test:np.ndarray, l:int, quadratic:bool = False) -> None:
        self.data = data
        self.data_test = data_test
        self.labels = labels
        self.l = l
        self.weights = None
        self.bias = None
        self.score_values = None
        self.is_fitted = False
        self.is_quadratic = quadratic

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def scores():
        pass

class logistic_regression(discriminative_model):
    def __init__(self, data:np.ndarray, labels:np.ndarray, data_test:np.ndarray, l:float = 0.1, quadratic:bool = False) -> None:
        super().__init__(data, labels, data_test, l, quadratic)

    def train(self):
        J = self.logreg_obj_wrap()
        x0 = np.zeros(self.data.shape[0] + 1)

        # print("data_size: " + str(self.data.shape))

        result_of_minimizer, _, _ = opt.fmin_l_bfgs_b(J, x0, approx_grad=True)
        # print("result of minimizer: " + str(result_of_minimizer))
        self.w = result_of_minimizer[:-1]
        self.b = result_of_minimizer[-1]
        # print("w: " + str(self.w))
        # print("b: " + str(self.b))
        self.is_fitted = True

    def scores(self):
        if(self.is_fitted is False):
            if (self.is_quadratic is True):
                self.quadratic_expansion()
            self.train()

        self.score_values = np.dot(self.w.T, self.data_test) + self.b
        # print("score_values_size : " + str(self.score_values.shape))
        # return np.dot(self.w.T, X) + self.b

    def logreg_obj_wrap(self):

        def logreg_obj(params:np.ndarray):
            # print("params: " + str(params))
            w, b = params[:-1], params[-1]
            z = 2 * self.labels - 1
            regularization_term = (self.l / 2) * np.dot(w, w.T)
            inner_sum = 0

            # print("size_z: " + str(z.shape))

            for i in range(self.data.shape[1]):
                # print("term: " + str((np.dot(w.T, self.data[:, i]) + b)))
                # print("size: " + str((np.dot(w.T, self.data[:, i]) + b).shape)) # should be an array of size 1
                # print("z[i]: " + str(z[i]))
                # print("i: " + str(i))
                second_term_of_log = -z[i] * (np.dot(w.T, self.data[:, i]) + b)
                inner_sum += np.logaddexp(0, second_term_of_log)
            
            value_to_return = regularization_term + inner_sum / self.data.shape[1]
            return value_to_return
        return logreg_obj
    
    def quadratic_expansion(self):
        # self.data = np.hstack((self.data, np.square(self.data)))
        # print("data_size before expansion: " + str(self.data[0, :].shape))
        # print("Before expansion:" + str(self.data))
        # self.data = np.hstack((self.data, self.data**2))
        # self.data = np.vstack((self.data, self.data[0, :]**2))
        self.data = np.vstack((self.data, self.data**2))
        self.data_test = np.vstack((self.data_test, self.data_test**2))
        # print("data_size after expansion: " + str(self.data[0, :].shape))
        # print("After expansion:" + str(self.data))
        


    # def vec(M):
    #     # return M.reshape(M.size, order='F')
    #     num_rows, num_cols = M.shape
    #     return M.reshape((num_rows * num_cols, 1))