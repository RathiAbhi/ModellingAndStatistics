#write the normal equation of linear regression used to determine the coefs
# X is the feature matrix and y is the target vector
import numpy as np

class NormalEquation:
    def normalEquation(self, X:list[list[float]],y:list[float]) -> list[float]:
        m = len(X)
        n = len(X[0])
        X = np.array(X)
        y = np.array(y)

        #calculating the transpose and then inverse
        Xt = X.reshape(n,m)
        A = np.matmul(Xt,X)
        B = np.matmul(Xt,y)
        C = np.linalg.inv(A)

        #calculate the final coef matrix
        theta = np.matmul(C,B)
        theta = np.round(theta,decimals=4)
        print(theta)
        return theta

    def main(self):
        if (np.allclose(self.normalEquation([[1, 1], [1, 2], [1, 3]], [1, 2, 3]), [-0.0, 1.0]) and
                np.allclose(self.normalEquation([[1, 3, 4], [1, 2, 5], [1, 3, 2]], [1, 2, 1]), [4.0, -1.0, 0.0])):
            print("pass")
        else:
            print("fail")

norm = NormalEquation()
norm.main()