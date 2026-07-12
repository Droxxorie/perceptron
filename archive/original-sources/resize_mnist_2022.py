import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


train = np.array(pd.read_csv('mnist_train.csv'))
test = np.array(pd.read_csv('mnist_test.csv'))

train_x = train[:,1:]
train_y = train[:,0]
test_x = test[:,1:]
test_y = test[:,0]

train_x = train_x.T / 255
test_x = test_x.T / 255

"""
Resize des images du MNIST via le plus proche voisin
"""

def resize14x14(x, h):
    """
    x est un matrice (n,m) avec n la dim de l'image
    """
    n,m = x.shape
    old_h = np.sqrt(n)
    xh = np.zeros((h * h, m))
    ratio = h / old_h

    for i in range(m):
        old_image = np.reshape(np.copy(x[:,i]), (28, 28))
        image = np.reshape(np.copy(xh[:,i]), (h, h))

        for k in range(h):
            for j in range(h):
                k_near = int(np.round(k / ratio))
                j_near = int(np.round(j / ratio))
                pixel = old_image[k_near, j_near]
                image[k, j] = pixel
        xh[:,i] = np.reshape(image, h * h)

    return xh

train_x14 = resize14x14(train_x, 14)
test_x14 = resize14x14(test_x, 14)

np.save('train_x14.npy', train_x14)
np.save('test_x14.npy', test_x14)

plt.figure()
plt.imshow(np.reshape(train_x[:,0], (28, 28)))
plt.show()
plt.figure()
plt.imshow(np.reshape(train_x14[:,0], (14, 14)))
plt.show()