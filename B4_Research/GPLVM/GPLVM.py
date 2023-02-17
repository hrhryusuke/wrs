import numpy as np

class GPLVM(object):
    def __init__(self,Y,LatentDim,HyperParam,X=None):
        self.Y = Y
        self.hyperparam = HyperParam
        self.dataNum = self.Y.shape[0]
        self.dataDim = self.Y.shape[1]

        self.latentDim = LatentDim
        if X is not None:
            self.X = X
        else:
            self.X = 0.1*np.random.randn(self.dataNum,self.latentDim)
        self.S = Y @ Y.T
        self.history = {}

    def fit(self,epoch=100,epsilonX=0.5,epsilonSigma=0.0025,epsilonAlpha=0.00005):

        resolution = 10
        M = resolution**self.latentDim
        self.history['X'] = np.zeros((epoch, self.dataNum, self.latentDim))
        self.history['F'] = np.zeros((epoch, M, self.dataDim))
        sigma = np.log(self.hyperparam[0])
        alpha = np.log(self.hyperparam[1])
        for i in range(epoch):

            # 潜在変数の更新
            K = self.kernel(self.X,self.X,self.hyperparam[0]) + self.hyperparam[1]*np.eye(self.dataNum)
            Kinv = np.linalg.inv(K)
            G = 0.5*(Kinv @ self.S @ Kinv-self.dataDim*Kinv)
            dKdX = -2.0*(((self.X[:,None,:]-self.X[None,:,:])*K[:,:,None]))/self.hyperparam[0]
            # dFdX = (G[:,:,None] * dKdX).sum(axis=1)-self.X
            dFdX = (G[:,:,None] * dKdX).sum(axis=1)

            # ハイパーパラメータの更新
            Dist = ((self.X[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2)
            dKdSigma = 0.5*Dist/self.hyperparam[0] * ( K- self.hyperparam[1]*np.eye(self.dataNum) )
            dFdSigma = np.trace(G @ dKdSigma)

            dKdAlpha = self.hyperparam[1]*np.eye(self.dataNum)
            dFdAlpha = np.trace(G @ dKdAlpha)

            self.X = self.X + epsilonX * dFdX
            self.history['X'][i] = self.X
            sigma = sigma + epsilonSigma * dFdSigma
            self.hyperparam[0] = np.exp(sigma)
            alpha = alpha + epsilonAlpha * dFdAlpha
            self.hyperparam[1] = np.exp(alpha)

            zeta = np.meshgrid(np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), resolution),
                               np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), resolution))
            zeta = np.dstack(zeta).reshape(M, self.latentDim)
            K = self.kernel(self.X,self.X,self.hyperparam[0]) + self.hyperparam[1]*np.eye(self.dataNum)
            Kinv = np.linalg.inv(K)
            KStar = self.kernel(zeta, self.X, self.hyperparam[0])
            self.F = KStar @ Kinv @ self.Y
            self.history['F'][i] = self.F


    def kernel(self,X1, X2, length):
        Dist = (((X1[:, None, :] - X2[None, :, :]) ** 2) / length).sum(axis=2)
        K = np.exp(-0.5 * Dist)
        return K
