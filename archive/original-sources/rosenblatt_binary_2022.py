import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pandas as pd
from matplotlib import animation
import time

"""
On va ici traiter le cas du perceptron de Rosenblatt pour les données du MNIST de 0 à 1
"""

###############################################################################

train = np.array(pd.read_csv('mnist_train.csv'))
test = np.array(pd.read_csv('mnist_test.csv'))

train_x = train[:,1:]
train_y = train[:,0]
test_x = test[:,1:]
test_y = test[:,0]

train_x = train_x / 255
test_x = test_x / 255

indice_train_0 = (train_y == 0)
indice_train_1 = (train_y == 1)
indice_train_01 = indice_train_0 + indice_train_1

train_x01 = train_x[indice_train_01].T
train_y01 = train_y[indice_train_01].T

indice_test_0 = (test_y == 0)
indice_test_1 = (test_y == 1)
indice_test_01 = indice_test_0 + indice_test_1

test_x01 = test_x[indice_test_01].T
test_y01 = test_y[indice_test_01].T

for i in range(len(train_y01)):
    if train_y01[i] == 0:
        train_y01[i] = -1

for i in range(len(test_y01)):
    if test_y01[i] == 0:
        test_y01[i] = -1

##################################### Fonctions ##########################################

def signum(x):
    """
    fonction d'activation, renvoie le signe de l'input (donc -1 ou 1)
    """
    for i in range(len(x)):
        if x[i] >= 0:
            x[i] = 1
        else:
            x[i] = -1
    return x

def init_param(x):
    """
    Initialisation des poids selon une loi uniforme
    """
    sizex, m = x.shape

    w = rnd.uniform(-1, 1, sizex)
    b = 0 #biais inutile

    return w, b

def precision(y, sigma):
    """
    Compare l'output avec y et renvoie la part de classes identiques
    """
    acc = np.sum(sigma == y) / len(y)

    return acc

def forwardprop(x, y, w):
    z = np.dot(w, x)
    sigma = signum(z)

    for j in range(len(x)):
        if sigma[j] * y[j] <= 0:
            w = w - (lrate * sigma[j] * x[:,j]).T
            w = w / np.linalg.norm(w)

    return w, sigma

def confusion_matrix(sigma, y):
    """
    Construit la matrice de confusion des classes de sortie par rapport aux réels
    """
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(len(y)):
        if sigma[i] == 1 and y[i] == 1:
            TP += 1
        if sigma[i] == 1 and y[i] == -1:
            FP += 1
        if sigma[i] == -1 and y[i] == -1:
            TN += 1
        if sigma[i] == -1 and y[i] == 1:
            FN += 1
    
    mat = np.array([[0, 0], [0, 0]])
    mat[0, 0] = TN
    mat[0, 1] = FP
    mat[1, 0] = FN
    mat[1, 1] = TP

    return mat

def perceptron_entrainement_Rosenblatt(x, y, iterations=100, pas=10):
    """
    Application de l'algorithme du perceptron de Rosenblatt sur les données de test
    renvoie la matrice des poids permetant de determiner la classe (-1 ou 1) d'une image carré
    """
    acc = []
    w, b = init_param(x) 
    wn = np.linalg.norm(w)
    R = np.max(np.linalg.norm(x))
    gamma = np.max(y*np.dot(w, x) / wn)
    n = int((R / gamma) ** 2)
    temps = time.time()
    t = time.time()
    print(f"Nombre maximal d'erreurs = {n}")

    for i in range(iterations):
        w, sigma = forwardprop(x, y, w)
        acc.append(precision(y, sigma))

        if (i + 1) % pas == 0:
            print(f'| Itération : {i + 1:4d}/{iterations} | Précision : {(precision(y, sigma) * 100):.2f}% | Temps : {(time.time() - t):.2f} s |')
            t = time.time()

    print(f'Précision entrainement : {precision(y, sigma) * 100:.2f}%')
    print(f"Temps d'entrainement : {time.time() - temps:.2f}s")

    return w, b, acc

def Perceptron_Rosenblatt(xtrain, ytrain, xtest, ytest, iterations=100, pas=10):
    """
    On applique les poids trouvées lors de l'entrainement au set de test et on regarde la précision
    """
    w, b, acc = perceptron_entrainement_Rosenblatt(xtrain, ytrain, iterations, pas)

    z = np.dot(w, xtest)
    sigma = signum(z)
    print(f'Précision test : {precision(ytest, sigma) * 100:.2f}%')
    return w, acc, sigma

def plot_img(xtrain, ytrain, xtest, ytest, iterations=100, pas=10):
    """
    On plot l'évolution de la précision d'entrainement et de test
    L'image de la matrice de confusion
    L'image de la matrice des poids
    Plot des éléments de la matrice de confusion
    """
    w, b = init_param(xtrain)
    TP = []
    FN = []
    FP = []
    TN = []
    n = np.arange(0, iterations, 1)
    acctest = []
    acc = []

    for i in range(iterations):
        w, sigma = forwardprop(xtrain, ytrain, w)
        mat2 = confusion_matrix(sigma, ytrain)
        TP.append(mat2[0,0])
        FN.append(mat2[1,0])
        FP.append(mat2[0,1])
        TN.append(mat2[1,1])
        z = np.dot(w, xtest)
        sigmatest = signum(z)
        acctest.append(precision(ytest, sigmatest))
        acc.append(precision(ytrain, sigma))

    z = np.dot(w, xtest)
    sigmatest = signum(z)

    mat = confusion_matrix(sigmatest, ytest)

    fig, ax = plt.subplots()
    plt.imshow(mat, cmap='Blues')
    ax.set_xticks(ticks=[0,1], labels=('Vrai 0', 'Vrai 1'))
    ax.set_yticks(ticks=[0,1], labels=('Prédit 0', 'Prédit 1'))
    ax.text(0, 0, mat[0,0], ha="center", va="center", color="w", size=20)
    ax.text(1, 0, mat[1,0], ha="center", va="center", color="w", size=20)
    ax.text(0, 1, mat[0,1], ha="center", va="center", color="w", size=20)
    ax.text(1, 1, mat[1,1], ha="center", va="center", color="w", size=20)
    plt.colorbar()
    plt.title(f"Matrice de confusion d'entrainement\n lrate = {lrate}, Itérations : {iterations}")
    plt.show()
  
    plt.figure()
    plt.plot(n, TP, label='TP')
    plt.plot(n, TN, label='TN')
    plt.plot(n, FP, label='FP')
    plt.plot(n, FN, label='FN')
    plt.legend()
    plt.title('Plot des élément de la matrice de confusion')
    plt.xlabel('Itérations')
    plt.ylabel("Nombre d'images")
    plt.show()

    img = np.copy(w)
    img.resize((28, 28))
    plt.figure()
    plt.title(f"Matrices des poids d'entrainement \n lrate = {lrate}, Itérations : {iterations}")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    plt.figure()
    plt.title(f"Évolution de la précision, lrate = {lrate}")
    plt.xlabel('Itérations')
    plt.ylabel('Précision')
    plt.plot(np.arange(0, iterations), acc, 'b', label='Entrainement')
    plt.plot(np.arange(0, iterations), acctest, 'r--', label='Test')
    plt.legend()
    plt.show()

    return

def predire_01(n, iterations=100):
    """
    On applique les poids à une image du set d'entrainement et on regarde la prédiction et la réalité
    """
    w, b = init_param(train_x01) 

    for i in range(iterations):
        w, sigma = forwardprop(train_x01, train_y01, w)

    z2 = np.dot(w, test_x01[:,n])
    z3 = 0

    if z2 <= 0:
        z2 = 0
    else:
        z2 = 1

    if test_y01[n] <= 0:
        z3 = 0
    else:
        z3 = 1

    #print(f'Prédiction : {z2} \n')
    #print(f'Attendu : {z3}')

    img = test_x01[:,n]
    plt.figure()
    plt.title(f'Prédiction : {z2} \nAttendu : {z3}')
    plt.imshow(img.reshape((28, 28)), cmap='gray')
    plt.axis('off')
    plt.show()

    return

def animationw(x, y, iterations=100):
    """
    animation de l'évolution des valeurs de la matrice des poids d'entrainement
    """
    w, b = init_param(x)
    myimages = []
    fig = plt.figure()
    plt.axis('off')

    for i in range(iterations):
        img = np.copy(w)
        img.resize((28, 28))
        imgplot = plt.imshow(img)
        myimages.append([imgplot])

        w, sigma = forwardprop(x,y,w)
    my_anim = animation.ArtistAnimation(fig, myimages)
    return my_anim

def animation_confusion(x, y, iterations=100, pas=10):
    """
    animation de l'évolution des valeurs de la matrice de confusion d'entrainement
    """
    w, b = init_param(x)
    myimages = []
    fig, ax = plt.subplots()

    for i in range(iterations):
        w, sigma = forwardprop(x, y, w)
        mat = confusion_matrix(sigma, y)
        img = np.copy(mat)
        imgpmat = ax.imshow(img) 
        ax.set_xticks(ticks=[0,1], labels=('Vrai 0','Vrai 1'))
        ax.set_yticks(ticks=[0,1], labels=('Prédit 0','Predit 1'))
        myimages.append([imgpmat])

    my_anim = animation.ArtistAnimation(fig, myimages)

    return my_anim

def animation_boudarie(x, y, iterations=100):
    """
    On cherche ici à regarder comment l'hyperplan des poids divise l'espace des classes
    on passe le problème en 2d
    on chosit deux couples de coordonnées un pour les 0 et un pour les 1
    le pixel au centre de l'image est minimal pour les 0 et maximal pour les 1
    les pixels des collones 6,7,8 ont une valeur minimal pour 1 et maximal pour les 0
    on affiche ces deux pixels pour toutes les images ainsi que la droite formée par les poids associés
    """
    m = 5000
    w, b = init_param(train_x01)
    x1 = np.ones(m)
    x0 = np.ones(m)
    y0 = rnd.normal(0, size=m)
    fig = plt.figure()
    myimages = []

    for i in range(m):
        x1[i] = x[435,i] * y[i]
        x0[i] = x[412,i] * y[i]

    for i in range(iterations):
        w01 = [w[435], w[412]]
        w, sigma = forwardprop(x, y, w)
        plot = plt.plot([-1,1], w01, 'r')
        myimages.append(plot)

    plt.plot(x0, x1, 'b,')
    my_anim = animation.ArtistAnimation(fig, myimages)

    return my_anim

def plotlrate(xtrain,ytrain):
    """
    plot de l'évolution de la précision d'entrainement à différentes lrate
    """
    w0, b = init_param(xtrain)
    w1, b = init_param(xtrain)
    w2, b = init_param(xtrain)
    w3, b = init_param(xtrain)
    w4, b = init_param(xtrain)
    acc0 = []
    acc1 = []
    acc2 = []
    acc3 = []
    acc4 = []
    n = np.arange(0, 1000, 1)
    lamda = [0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
    
    for i in range(1000):
        z0 = np.dot(w0, xtrain)
        sigma0 = signum(z0)
        z1 = np.dot(w1, xtrain)
        sigma1 = signum(z1)
        z2 = np.dot(w2, xtrain)
        sigma2 = signum(z2)
        z3 = np.dot(w3, xtrain)
        sigma3 = signum(z3)
        z4 = np.dot(w4, xtrain)
        sigma4 = signum(z4)
    
        for j in range(len(xtrain)):
            if sigma0[j] * ytrain[j] <= 0:
                w0 = w0 - (lamda[0] * sigma0[j] * xtrain[:,j]).T
                w0 = w0 / np.linalg.norm(w0)

            if sigma1[j] * ytrain[j] <= 0:
                w1 = w1 - (lamda[1] * sigma1[j] * xtrain[:,j]).T
                w1 = w1 / np.linalg.norm(w1)

            if sigma2[j] * ytrain[j] <= 0:
                w2 = w2 - (lamda[2] * sigma2[j] * xtrain[:,j]).T
                w2 = w2 / np.linalg.norm(w2)

            if sigma3[j] * ytrain[j] <= 0:
                w3 = w3 - (lamda[3] * sigma3[j] * xtrain[:,j]).T
                w3 = w3 / np.linalg.norm(w3)

            if sigma4[j] * ytrain[j] <= 0:  
                w4 = w4 - (lamda[4] * sigma4[j] * xtrain[:,j]).T
                w4 = w4 / np.linalg.norm(w4)

        acc0.append(precision(ytrain, sigma0))
        acc1.append(precision(ytrain, sigma1))
        acc2.append(precision(ytrain, sigma2))
        acc3.append(precision(ytrain, sigma3))
        acc4.append(precision(ytrain, sigma4))

    plt.figure()
    plt.plot(n, acc0, label=f'{lamda[0]}')
    plt.plot(n, acc1, label=f'{lamda[1]}')
    plt.plot(n, acc2, label=f'{lamda[2]}')
    plt.plot(n, acc3, label=f'{lamda[3]}')
    plt.plot(n, acc4, label=f'{lamda[4]}')
    plt.xlabel('Itérations')
    plt.ylabel('Précision')
    plt.title('Précision à différents lrate')
    plt.legend()
    plt.show()

    return

################################# Main ##############################################   

lrate = 0.0001

#Perceptron_Rosenblatt(train_x01,train_y01,test_x01,test_y01,iterations=100,pas=100)
#plot_img(train_x01,train_y01,test_x01,test_y01,iterations=100,pas=10)
#predire_01(10,100)

#my_anim = animationw(train_x01,train_y01,iterations=100)
#my_anim.save('animation.gif', fps=20)
#plt.show(my_anim)

#my_anim = animation_confusion(train_x01,train_y01,iterations=100)
#my_anim.save('animation_confusion01.gif', fps=20)
#plt.show(my_anim)

#my_anim = animation_boudarie(train_x01,train_y01,iterations=100)
#my_anim.save('animation_boudarie.gif', fps=1)
#plt.show(my_anim)

plotlrate(train_x01, train_y01)

