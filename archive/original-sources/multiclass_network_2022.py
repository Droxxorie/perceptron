import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pandas as pd
from matplotlib import animation
import time

"""
On va ici traiter le cas du perceptron multilayered pour les données complètes du MNIST (chiffres manuscrits de 0 à 9)
"""

###############################################################################

train = np.array(pd.read_csv('mnist_train.csv'))
test = np.array(pd.read_csv('mnist_test.csv'))

train_x = train[:,1:]
train_y = train[:,0]
test_x = test[:,1:]
test_y = test[:,0]

train_x = train_x.T/255
test_x = test_x.T/255

train_x14 = np.load('train_x14.npy')
test_x14 = np.load('test_x14.npy')

##################################### Fonctions ##########################################

def ReLu(z):
    """
    Fonction d'activation, Rectified Linear Unit
    Permet un descente de gradiant, la dérivée n'est pas constante
    N'active pas tous les neuronnes et est donc plus efficace
    Dying ReLu
    """
    return np.maximum(z,0)

def dReLu(z):
    """
    Dérivée de la fonction ReLU
    """
    return z > 0

def leaky_ReLu(z):
    """
    Fonction d'activation de même type que ReLu
    Corrige le dying ReLu
    """
    return np.maximum(z,0.1*z)

def dleaky_ReLu(z):
    a = z >= 0
    b = z < 0
    return a + 0.1*b

def sigmoid(z):
    """
    Fonction d'activation
    Rends des valeurs entre 0 et 1
    z est 
    """
    one = np.ones_like(z)
    return one/(one + np.exp(-z))

def dsigmoid(z):
    """
    Dérivées de sigmoid
    """
    one = np.ones_like(z)
    return np.exp(-z)/(one + np.exp(-z))**2

def softmax(z):
    """
    Une fonction de sortie qui permet d'avoir des probabilités en sortie (la somme des éléments des collones sera 1)
    """
    return np.exp(z)/sum(np.exp(z))

def onehot(y):
    """
    Fonction qui attribu un 1 à la position de la valeur (de 0 à 10) de chaque élément de y dans une matrice 10*m
    """
    m = y.size
    yoh = np.zeros((10,m))
    
    for i in range(m):
       yoh[y[i],i] = 1

    return yoh

def init_param(x):
    """
    Initialise aléatoirement les poids et les biais entre -1 et 1 via une loi uniforme 
    """
    sizex, m = x.shape
    
    w1 = rnd.uniform(-1,1,(10,sizex))
    b1 = rnd.uniform(-1,1)*np.ones((10,1))
    w2 = rnd.uniform(-1,1,(10,10))
    b2 = rnd.uniform(-1,1)*np.ones((10,1))

    return w1, b1, w2, b2

def prediction(sigma2):
    """
    Prends un matrice sigma2 avec des éléments de 0 à 1 et rends une matrice avec les indices des valeurs maximum de chaque collones
    """
    prediction = np.argmax(sigma2,0)
    return prediction

def precision(y,sigma2):
    """
    Compare la prédiction à la réalité et rends le pourcentage de bonnes prédictions
    """
    precision = np.sum(prediction(sigma2) == y)/len(y)
    return precision
    
def forwardprop(x,w1,b1,w2,b2):
    """
    x l'input : une matrice 10*m
    w1 les poids pour l'input : une matrice 10*m
    b1 le biais pour les images 
    w2 les poids pour la couche intérmédiaire : une marice 10*10
    b2 les poids pour la couche intermédiaire
    
    Fonction qui applique les poids et les biais pour la couche intérmédiaire et l'output
    """
    z1 = np.dot(w1,x) + b1
    #sigma1 = ReLu(z1)
    #sigma1 = leaky_Relu(z1)
    sigma1 = sigmoid(z1)
    
    z2 = np.dot(w2,sigma1) + b2
    sigma2 = softmax(z2)
    
    return z1, sigma1, z2, sigma2
  
def backwardprop(x,y,z1,sigma1,z2,sigma2,w2):
    """
    dérivée des élément selon sigma2
    dz2 est la dérivée de la variance*1/2
    """
    m = y.size

    dz2 = (sigma2 - onehot(y))
    dw2 = np.dot(dz2,sigma1.T)/m
    db2 = np.sum(dz2,axis=1,keepdims=True)/m

    #dz1 = np.dot(w2.T,dz2)*dReLu(z2)
    #dz1 = np.dot(w2.T,dz2)*dleaky_ReLu(z2)
    dz1 = np.dot(w2.T,dz2)*dsigmoid(z2)
    dw1 = np.dot(dz1,x.T)/m
    db1 = np.sum(dz1,axis=1,keepdims=True)/m
    
    return dz2, dw1, db1, dw2, db2

def descente_grad(w1,b1, w2, b2, dw1, db1, dw2, db2, lrate):
    """
    Corréction des poids et des biais via une descente de gradient
    """
    w1 = w1 - lrate*dw1
    b1 = b1 - lrate*db1

    w2 = w2 - lrate*dw2
    b2 = b2 - lrate*db2

    return w1, b1, w2, b2

def Perceptron_entrainement(x, y, iterations, lrate, pas=10):
    """
    Application de l'algorithme du Perceptron avec une couche intérmédiaire
    On cherche les poids et les biais tels que avec un x inconnu la précision sur y soit maximale (ou que la variance soit minimale)
    1. Initialisation des poids et biais
    2. Applique les biais et poids à l'input x
    3. Descente de gradient sur chaque éléments
    4. On revient à l'étape 1 et on continue jusqu'au nombre d'itérations
    """
    temps = time.time()
    t = time.time()
    
    w1, b1, w2, b2 = init_param(x)
    
    print(f'lrate : {lrate}')
    for i in range(iterations):
        
        z1, sigma1, z2, sigma2 = forwardprop(x,w1,b1,w2,b2)
        dz2, dw1, db1, dw2, db2 = backwardprop(x,y,z1,sigma1,z2,sigma2,w2)
        w1, b1, w2, b2 = descente_grad(w1,b1, w2, b2, dw1, db1, dw2, db2, lrate)
        
        if (i + 1) % pas == 0:
            print(f'| Itération : {i+1:4d}/{iterations} | Précision : {(precision(y,sigma2)*100):.2f}% | Temps : {(time.time() - t):.2f} s |')
            t = time.time()
    print(f'Précision entrainement : {precision(y,sigma2)*100:.2f}%')
    print(f"Temps d'entrainement : {time.time() - temps:.2f} s")
    return w1, b1, w2, b2

def Perceptron_test(xtrain,ytrain,xtest,ytest,iterations,lrate,pas):
    """
    On applique les poids trouvés lors de l'entrainement à une matrice x inconnue
    et on compare à la matrice y réel
    """
    w1, b1, w2, b2 = Perceptron_entrainement(xtrain,ytrain,iterations,lrate,pas)
    sigma2 = forwardprop(xtest,w1,b1,w2,b2)[3]
    
    print(f'Précision test : {precision(ytest,sigma2)*100:.2f}%')
    
    return sigma2

def test_predire(w1, b1, w2, b2):
    """
    On choisit 4 nombres aléatoires et on regarde les images à ces indices ainsi que le chiffre prédit et le chiffre réel
    """
    n = rnd.randint(0,100,size=4)
    sigma2 = forwardprop(test_x,w1,b1,w2,b2)[3]
    prdct = prediction(sigma2)
    
    plt.figure()
    
    ax1 = plt.subplot(221)
    ax1.imshow(np.reshape(test_x[:,n[0]],(28,28)),cmap='gray')
    ax1.set_title(f'Prédit : {prdct[n[0]]} Réel : {test_y[n[0]]}')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    
    ax2 = plt.subplot(222)
    ax2.imshow(np.reshape(test_x[:,n[1]],(28,28)),cmap='gray')
    ax2.set_title(f'Prédit : {prdct[n[1]]} Réel : {test_y[n[1]]}')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    
    ax3 = plt.subplot(223)
    ax3.imshow(np.reshape(test_x[:,n[2]],(28,28)),cmap='gray')
    ax3.set_title(f'Prédit : {prdct[n[2]]} Réel : {test_y[n[2]]}')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    
    ax4 = plt.subplot(224)
    ax4.imshow(np.reshape(test_x[:,n[3]],(28,28)),cmap='gray')
    ax4.set_title(f'Prédit : {prdct[n[3]]} Réel : {test_y[n[3]]}')
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    
    plt.subplots_adjust(wspace=-0.4, hspace=0.3)
    plt.show()
    
    return

def confusion_matrix(pred,y):
    """
    Matrice de confusion de pour l'entrainement
    les vrai positifs sont la diagonale principale
    """
    mat = np.zeros((10,10))

    for i in range(len(y)):
        mat[y[i],pred[i]] = mat[y[i],pred[i]] + 1

    return mat

def plot_image(x,y,xtest,ytest,iterations, lrate):
    """
    Plot de la précision et de la variance au cours des itérations pour l'entrainement
    représentation visuelle de la matrice de confusion pour l'entrainement
    Image des matrices de poids d'input
    Image de la matrices des poids de la couche intermédiaire
    """
    n,m = x.shape
    ntest,mtest=xtest.shape
    n = int(np.sqrt(n))
    w1, b1, w2, b2 = init_param(x)
    precision_train = []
    precision_test = []
    var = []
    vartest = []
    I = np.arange(0,iterations,1)
    
    
    for i in range(iterations):
        
        z1, sigma1, z2, sigma2 = forwardprop(x,w1,b1,w2,b2)
        sigma2test = forwardprop(xtest,w1,b1,w2,b2)[3]
        dz2, dw1, db1, dw2, db2 = backwardprop(x,y,z1,sigma1,z2,sigma2,w2)
        w1, b1, w2, b2 = descente_grad(w1,b1, w2, b2, dw1, db1, dw2, db2, lrate)
        precision_train.append(precision(y,sigma2)*100)
        precision_test.append(precision(ytest,sigma2test)*100)
        var.append(np.sum(dz2**2)/m)
        vartest.append(np.sum((sigma2test - onehot(ytest))**2)/mtest)
        
    plt.figure()
    plt.plot(I,precision_train,label='Entrainement')
    plt.plot(I,precision_test,'r--',label='Test')
    plt.xlabel('Itérations')
    plt.legend()
    plt.title(f"Évolution de la précision, lrate = {lrate} ({n}x{n})")
    plt.show()
    
    plt.figure()
    plt.plot(I,var,label='Entrainement')
    plt.plot(I,vartest,'r--',label='Test')
    plt.xlabel('Itérations')
    plt.title(f"Évolution de la variance, lrate = {lrate} ({n}x{n})")
    plt.legend()
    plt.show()
    
    mat1 = confusion_matrix(prediction(sigma2),y)
    plt.figure()
    plt.imshow(mat1,cmap='Blues')
    plt.xticks(np.arange(0, 10), ['0', '1', '2', '3', '4','5','6','7','8','9'])
    plt.yticks(np.arange(0, 10), ['0', '1', '2', '3', '4','5','6','7','8','9'])
    plt.colorbar()
    plt.title(f"Matrice de confusion d'entrainement\n lrate = {lrate}, Itérations : {iterations} ({n}x{n})")
    plt.xlabel('Prédits')
    plt.ylabel('Réels')
    plt.show()
    
    fig, axs = plt.subplots(4,3,figsize=(24,14))
    fig.suptitle(f"Matrices des poids d'entrée \n lrate = {lrate}, Itérations : {iterations} ({n}x{n})",fontsize=20)
    
    for i in range(4):
        for j in range(3):
            axs[i,j].get_xaxis().set_visible(False)
            axs[i,j].get_yaxis().set_visible(False)
    
    axs[0,0].imshow(np.reshape(w1[0],(n,n)))
    axs[0,1].imshow(np.reshape(w1[1],(n,n)))
    axs[0,2].imshow(np.reshape(w1[2],(n,n)))
    axs[1,0].imshow(np.reshape(w1[3],(n,n)))
    axs[1,1].imshow(np.reshape(w1[4],(n,n)))
    axs[1,2].imshow(np.reshape(w1[5],(n,n)))
    axs[2,0].imshow(np.reshape(w1[6],(n,n)))
    axs[2,1].imshow(np.reshape(w1[7],(n,n)))
    axs[2,2].imshow(np.reshape(w1[8],(n,n)))
    axs[3,1].imshow(np.reshape(w1[9],(n,n)))
    axs[3,0].set_visible(False)
    axs[3,2].set_visible(False)
    
    axs[0,0].set_title('0',fontsize=20)
    axs[0,1].set_title('1',fontsize=20)
    axs[0,2].set_title('2',fontsize=20)
    axs[1,0].set_title('3',fontsize=20)
    axs[1,1].set_title('4',fontsize=20)
    axs[1,2].set_title('5',fontsize=20)
    axs[2,0].set_title('6',fontsize=20)
    axs[2,1].set_title('7',fontsize=20)
    axs[2,2].set_title('8',fontsize=20)
    axs[3,1].set_title('9',fontsize=20)
    
    plt.subplots_adjust(wspace=-0.8, hspace=0.2)
    plt.show()
    
    plt.figure()
    plt.title(f'Matrice des poids intermédiaire\n lrate = {lrate}, Itérations : {iterations} ({n}x{n})')
    plt.axis('off')
    plt.imshow(w2)
    plt.colorbar()
    plt.show()
    
    return
        
def plot_rate(x,y,m):
    h = int(np.sqrt(x.shape[0]))
    """
    plot de la précision lors de l'entrainement à différents lrate
    """
    w11, b11, w21, b21 = init_param(x)
    w12, b12, w22, b22 = init_param(x)
    w13, b13, w23, b23 = init_param(x)
    w14, b14, w24, b24 = init_param(x)
    w15, b15, w25, b25 = init_param(x)
    acc1 = []
    acc2 = []
    acc3 = []
    acc4 = []
    acc5 = []
    var1 = []
    var2 = []
    var3 = []
    var4 = []
    var5 = []
    
    n = np.arange(0,m,1)
    lamda = [0.6,0.4,0.2,0.1,0.01]
    
    for i in range(m):
        z11, sigma11, z21, sigma21 = forwardprop(x,w11,b11,w21,b21)
        dz21, dw11, db11, dw21, db21 = backwardprop(x,y,z11,sigma11,z21,sigma21,w21)
        w11, b11, w21, b21 = descente_grad(w11,b11, w21, b21, dw11, db11, dw21, db21, lamda[0])
        acc1.append(precision(y,sigma21)*100)
        var1.append(np.sum(dz21**2)/len(y))
        
        z12, sigma12, z22, sigma22 = forwardprop(x,w12,b12,w22,b22)
        dz22, dw12, db12, dw22, db22 = backwardprop(x,y,z12,sigma12,z22,sigma22,w22)
        w12, b12, w22, b22 = descente_grad(w12,b12, w22, b22, dw12, db12, dw22, db22, lamda[1])
        acc2.append(precision(y,sigma22)*100)
        var2.append(np.sum(dz22**2)/len(y))
        
        z13, sigma13, z23, sigma23 = forwardprop(x,w13,b13,w23,b23)
        dz23, dw13, db13, dw23, db23 = backwardprop(x,y,z13,sigma13,z23,sigma23,w23)
        w13, b13, w23, b23 = descente_grad(w13,b13, w23, b23, dw13, db13, dw23, db23, lamda[2])
        acc3.append(precision(y,sigma23)*100)
        var3.append(np.sum(dz23**2)/len(y))
        
        z14, sigma14, z24, sigma24 = forwardprop(x,w14,b14,w24,b24)
        dz24, dw14, db14, dw24, db24 = backwardprop(x,y,z14,sigma14,z24,sigma24,w24)
        w14, b14, w24, b24 = descente_grad(w14,b14, w24, b24, dw14, db14, dw24, db24, lamda[3])
        acc4.append(precision(y,sigma24)*100)
        var4.append(np.sum(dz24**2)/len(y))
        
        z15, sigma15, z25, sigma25 = forwardprop(x,w15,b15,w25,b25)
        dz25, dw15, db15, dw25, db25 = backwardprop(x,y,z15,sigma15,z25,sigma25,w25)
        w15, b15, w25, b25 = descente_grad(w15,b15, w25, b25, dw15, db15, dw25, db25, lamda[4])
        acc5.append(precision(y,sigma25)*100)
        var5.append(np.sum(dz25**2)/len(y))
    
    plt.figure()
    plt.title(f'Évolution de la précision ({h}x{h})')
    plt.xlabel('Itérations')
    plt.ylabel('Précision')
    plt.plot(n,acc1,label=f'{lamda[0]}')
    plt.plot(n,acc2,label=f'{lamda[1]}')
    plt.plot(n,acc3,label=f'{lamda[2]}')
    plt.plot(n,acc4,label=f'{lamda[3]}')
    plt.plot(n,acc5,label=f'{lamda[4]}')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title(f'Évolution de la variance ({h}x{h})')
    plt.xlabel('Itérations')
    plt.ylabel('Variance')
    plt.plot(n,var1,label=f'{lamda[0]}')
    plt.plot(n,var2,label=f'{lamda[1]}')
    plt.plot(n,var3,label=f'{lamda[2]}')
    plt.plot(n,var4,label=f'{lamda[3]}')
    plt.plot(n,var5,label=f'{lamda[4]}')
    plt.legend()
    plt.show()
    
    return

def animation_confusion(x,y,iterations,lrate):
    """
    Crée un gif de l'évolution de la matrice de confusion pour l'entrainenemnt
    """
    n = int(np.sqrt(x.shape[0]))
    w1, b1, w2, b2 = init_param(x)
    myimages = []
    
    fig, ax = plt.subplots()
    ax.set_title(f"Matrice de confusion d'entrainement\n lrate = {lrate}, Itérations : {iterations} ({n}x{n})")
    plt.xticks(np.arange(0, 10), ['0', '1', '2', '3', '4','5','6','7','8','9'])
    plt.yticks(np.arange(0, 10), ['0', '1', '2', '3', '4','5','6','7','8','9'])
    plt.xlabel('Prédits')
    plt.ylabel('Réels')
    
    for i in range(iterations):
        
        z1, sigma1, z2, sigma2 = forwardprop(x,w1,b1,w2,b2)
        dz2, dw1, db1, dw2, db2 = backwardprop(x,y,z1,sigma1,z2,sigma2,w2)
        w1, b1, w2, b2 = descente_grad(w1,b1, w2, b2, dw1, db1, dw2, db2, lrate)
        acc = precision(y,sigma2)*100
        mat = confusion_matrix(prediction(sigma2),y)
        img = np.copy(mat)
        imgpmat = ax.imshow(img,cmap='Blues') 
        myimages.append([imgpmat])
    
    acc = precision(y,sigma2)*100
    
    my_anim = animation.ArtistAnimation(fig, myimages)
    ax.set_title(f"Matrice de confusion d'entrainement\n lrate = {lrate}, Itérations : {iterations} ({n}x{n}) {acc:.0f}%")
    return my_anim

################################### Main ############################################


#w1, b1, w2, b2 = Perceptron_entrainement(train_x,train_y,1000,0.2,100)
#test_predire(w1, b1, w2, b2)
#Perceptron_test(train_x,train_y,test_x,test_y,1000,0.1,100)
plot_image(train_x,train_y,test_x,test_y,100,0.1) 
#plot_rate(train_x14,train_y,10)

#my_anim = animation_confusion(train_x14,train_y,6000,0.04)
#my_anim.save('animation_confusion.gif', fps=60)
#plt.show(my_anim)