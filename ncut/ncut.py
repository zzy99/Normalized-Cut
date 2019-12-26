import numpy as np
import skimage
import skimage.io
from scipy import sparse
from scipy.sparse.linalg import eigs,svds
import matplotlib.pyplot as plt
from time import *

def generate_W(img):
    N = len(img)
    d = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        d[i, :] = np.abs(img - img[i])
    W = np.exp(-np.power(d / np.max(d), 2))
    return W

def n_cuts(W):
    D = np.sum(W, axis=1)
    D = np.diag(D)

    begin_time = time()

    s_D = sparse.csr_matrix(D)
    s_W = sparse.csr_matrix(W)
    s_D_nhalf = np.sqrt(s_D).power(-1)
    L = s_D_nhalf @ (s_D - s_W) @ s_D_nhalf
    # eigenvalues, eigenvectors = eigs(L,2,which='SM')
    # eigenvectors = np.transpose(eigenvectors)
    _, eigenvalues, eigenvectors = svds(L,which='SM')#svd much faster than eig(lanczos method)

    end_time = time()
    print('svd time: ',end_time-begin_time,'s',sep='')

    plt.figure()
    for i in range(5):
        Partition_1 = eigenvectors[i+1] > 0#choose 0 as split point, which can also be replaced by np.median(eigenvectors[i+1])
        plt.subplot(1, 5, i+1)
        # cut_area = image*Partition_1.reshape(image.shape)
        # skimage.io.imshow(cut_area)
        skimage.io.imshow(Partition_1.reshape(image.shape))
    plt.show()

if __name__ == '__main__':
    image = skimage.img_as_ubyte(skimage.io.imread('test.jpg',as_gray=True))#smaller than 100*100px better
    W = generate_W(image.flatten())
    n_cuts(W)

