import numpy as np

def hypothesis(x, theta):
  """
  Calculate the hypothesis function for every datapoint in x
  :param x: numpy array of size (n, d) where n is the number of samples
  and d is the number of features per sample including the 1 extra feature
  :param theta: numpy array of size (d,)
  :return: predicted probability.
  """
  theta_dot_x = theta.T @ x.T
  return theta_dot_x

class Solution():

    def compute_loss(w, X, Y):

        h_x = hypothesis(X, w)

        return np.sum(np.power(h_x - Y, 2).T, axis=0)


    def compute_gradient(w, X, Y):

        h_w = hypothesis(X, w)
        print(X)

        gradient = (2 * (h_w - Y) @ X)

        return gradient

if __name__ == '__main__':

    X = np.array([[0,0,0], [1,1,1]])
    Y = np.array([0,1])
    w = np.ones(len(X[0]))
    answer = Solution.compute_gradient(w, X, Y)
    print(np.allclose(answer , [4, 4, 4]))
