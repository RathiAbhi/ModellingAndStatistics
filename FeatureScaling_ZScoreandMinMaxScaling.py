import numpy as np

class Scaling:
    def scaling(self,X):
        mean = np.mean(X, axis=0)
        std = np.std(X,axis=0)
        standard = (X-mean)/std

        X_min = np.min(X,axis=0)
        X_max = np.max(X,axis=0)
        normal = (X-X_min)/(X_max - X_min)

        standard = np.append(standard,normal,axis=0)
        print(standard)
        return standard

    def main(self):
        X = np.array([[1,2],[3,4],[5,6]])
        self.scaling(X)

scale = Scaling()
scale.main()