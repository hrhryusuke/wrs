from GPLVM import GPLVM
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def createKuraData(N, D,sigma=0.1):
    X = (np.random.rand(N, 2) - 0.5) * 2
    Y = np.zeros((N, D))
    Y[:, :2] = X
    Y[:,2]=X[:,0]**2-X[:,1]**2
    Y += np.random.normal(0,sigma,(N,D))

    return [X,Y]

def plot_prediction(Y,f,Z,epoch,save_folder):
    fig = plt.figure(1,[10,8])

    nb_nodes = f.shape[1]
    nb_dim = f.shape[2]
    resolution = np.sqrt(nb_nodes).astype('int')
    for i in range(epoch):
        if i%10 is 0:
            ax_input = fig.add_subplot(1, 2, 1, projection='3d')
            ax_input.cla()
            ax_input.set_aspect("auto")
            # 観測空間の表示
            r_f = f[i].reshape(resolution, resolution, nb_dim)
            ax_input.plot_wireframe(r_f[:, :, 0], r_f[:, :, 1], r_f[:, :, 2],color='k')
            ax_input.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=Y[:, 0], edgecolors="k",marker='x')
            ax_input.set_xlim(Y[:, 0].min(), Y[:, 0].max())
            ax_input.set_ylim(Y[:, 1].min(), Y[:, 1].max())
            ax_input.set_zlim(Y[:, 2].min(), Y[:, 2].max())

            # 潜在空間の表示
            ax_latent = fig.add_subplot(1, 2, 2)
            ax_latent.cla()
            ax_latent.set_aspect("equal")
            ax_latent.set_xlim(Z[:,:, 0].min(), Z[:,:, 0].max())
            ax_latent.set_ylim(Z[:,:, 1].min(), Z[:,:, 1].max())
            ax_latent.scatter(Z[i,:, 0], Z[i,:, 1], c=Y[:, 0], edgecolors="k")
            plt.savefig(save_folder+"fig{0}.png".format(i))

        plt.pause(0.001)
    plt.show()

if __name__ == '__main__':
    L=2
    N=200
    D=3
    sigma=3
    alpha=0.05
    beta=0.08
    seedData=1
    resolution = 10
    M = resolution**L
    save_folder = "fig/"
    os.makedirs(save_folder,exist_ok=True)

    # 入力データの生成
    # np.random.seed(seedData)
    [X,Y] = createKuraData(N,D,sigma=0.01)

    # カーネルの設定
    [U,D,Vt] = np.linalg.svd(Y)
    model = GPLVM(Y,L, np.array([sigma**2,alpha/beta]))
    # GPLVMの最適化
    epoch = 200
    model.fit(epoch=epoch,epsilonX=0.05,epsilonSigma=0.0005,epsilonAlpha=0.00001)

    # 推定した潜在変数の取得
    X = model.history['X']
    f = model.history['F']

    # 学習結果の表示
    plot_prediction(Y,f,X,epoch,save_folder)
