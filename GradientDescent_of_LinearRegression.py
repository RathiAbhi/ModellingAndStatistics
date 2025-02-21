import numpy as np

class Gradient:
    def gradient(self,X,y,alpha,iterations):
        m,n = X.shape
        theta = np.zeros((n,1))
        y = y.reshape(-1,1)
        Xt = X.T
        for i in range(iterations):
            A = np.matmul(X,theta)
            B = np.subtract(A,y)
            const = (alpha/m)
            eff = np.matmul(Xt,B)
            eff = eff * const
            theta = np.subtract(theta,eff)

        res = np.round(theta.flatten(),4)
        print(res)
        return res

    def main(self):
        expected_output = np.array([0.1107, 0.9513])
        computed_output = self.gradient(np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000)

        if np.allclose(computed_output, expected_output):
            print("pass")
        else:
            print("fail")

grad = Gradient()
grad.main()