from keras.datasets import mnist
import numpy as np
import functions as fc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.stats
from sklearn.metrics import confusion_matrix

# Makes a list of c covariance matrices with 10 on the diagonal and 0 otherwise
def initialSigma(c, dimentions):

    # Make matrix with 10 on the diagonal
    sigma = np.zeros((dimentions, dimentions))
    for i in range(dimentions):
        sigma[i][i] = 10

    # Make c of those
    sigmas = np.zeros(((c, dimentions, dimentions)))
    for i in range(c):
        sigmas[i] = sigma

    return sigmas



# Using regularized EM to fit a gaussian mixture model for a given number.
# Plots log-likelihood and number of gaussians as a function of iterations.
def plotNumberLoglik(x_data, y_data, number, dimensions, c, iterationsCMeans, iterationsEM):

    #Extracts all the digits for a given number, and performs PCA
    index = np.where(y_data == number)
    x_train = x_data[index]
    pca = PCA(n_components=dimensions)
    pca.fit(x_train)
    print("Explained variance: ", np.sum(pca.explained_variance_ratio_))
    x_train_transformed = pca.transform(x_train)

    # Initializes the parameters using C-means algorithm
    centers, pis, sigmas = fc.cMeans(train=x_train_transformed, c=c, dim=dimensions,
                                     rangeMin=np.min(x_train_transformed),
                                     rangeMax=np.max(x_train_transformed), iterations=iterationsCMeans)

    # The sigmas that is returned from C-means doesn't work. The algorithm is therefore
    # initialized with diagonal matrices as sigma
    sigmas = initialSigma(c, dimensions)

    # Plots for all lambdas
    lambd = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

    loglik = []
    numberGauss = []

    # For each lambda
    for i in range(len(lambd)):

        # Perform EM-algorithm, and add to list
        newPis, newMys, newSigmas, newLoglik, newNumberGauss = fc.EMAlgorithm(x_train_transformed, pis, centers,
                                                                              sigmas, iter=iterationsEM,
                                                                              unknown_mys=True, unknown_sigmas=True,
                                                                              lambd=lambd[i])
        loglik.append(newLoglik)
        numberGauss.append(newNumberGauss)

    # Plots log-likelihood and number of gaussians as function of iterations
    iterations = np.linspace(1, iterationsEM, iterationsEM)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    for i in range(len(loglik)):
        plt.plot(iterations, loglik[i], label="Lambda = " + str(lambd[i]))
    plt.title("Log-likelihood for number " + str(number))
    plt.xlabel("Iterations")
    plt.ylabel("Log-likelihood")
    plt.legend()

    plt.subplot(122)
    for i in range(len(numberGauss)):
        plt.plot(iterations, numberGauss[i], label="Lambda = " + str(lambd[i]))
    plt.title("# of clusters for number " + str(number))
    plt.xlabel("Iterations")
    plt.ylabel("# of mixtures")
    plt.legend()
    plt.savefig("./Plots/Number" + str(number) + ".pdf")


# For each number: Performs PCA, and fits a gaussian mixture model, that can be used for predicting
def trainModel(x_train, y_train, lambd, dimensions, c, iterationsCMeans, iterationsEM):

    piList = []
    myList = []
    sigmaList = []
    pcaList = []

    # For each digit
    for i in range(10):

        print("Digit "+str(i))

        # Perform PCA
        index = np.where(y_train == i)
        x_numbers = x_train[index]
        pca = PCA(n_components=dimensions)
        pca.fit(x_numbers)
        pcaList.append(pca)
        x_train_transformed = pca.transform(x_numbers)

        # Initialize parameters with C-means algorithm
        centers, pis, sigmas = fc.cMeans(train=x_train_transformed, c=c[i], dim=dimensions,
                                         rangeMin=np.min(x_train_transformed),
                                         rangeMax=np.max(x_train_transformed), iterations=iterationsCMeans)

        # The sigmas from C-means doesn't work, uses diagonal matrices
        sigmas = initialSigma(c[i], dimensions)

        # Run EM algorithm
        newPis, newMys, newSigmas, newLoglik, newNumberGauss = fc.EMAlgorithm(x_train_transformed, pis, centers,
                                                                              sigmas, iter=iterationsEM,
                                                                              unknown_mys=True, unknown_sigmas=True,
                                                                              lambd=lambd)
        # Add resulting model to list for return
        piList.append(newPis)
        myList.append(newMys)
        sigmaList.append(newSigmas)

    return piList, myList, sigmaList, pcaList


# Takes as argument a fitted model for each digit, and predicts the class for test data
def predict(x_test, piList, myList, sigmaList, pcaList):

    # So-far most probable class with probabilities
    probabilities = np.zeros(len(x_test))
    y_predict = np.zeros(len(x_test))

    # For each digit
    for i in range(10):

        # Perform the same dimension reduction as we did on the test data when training the EM-model
        # (different for each digit)
        x_test_transformed = pcaList[i].transform(x_test)

        # Calculates the probability of each test sample being digit i
        newProbabilities = np.zeros(len(x_test))
        for j in range(len(piList[i])):
            newProbabilities += piList[i][j] * scipy.stats.multivariate_normal(myList[i][j],
                                        sigmaList[i][j]).pdf(x_test_transformed)

        # If probability of being digit i is bigger than previous best,
        # update currently best class prediction
        for j in range(len(newProbabilities)):
            if newProbabilities[j] > probabilities[j]:
                probabilities[j] = newProbabilities[j]
                y_predict[j] = i

    return y_predict


# Runs the function necessary to obtain the results in the report
# plotNumberLoglik produces the plots to analyse number of clusters for each digit
# trainModel trains model, and predict uses this model to classify the test data.
# Finally the confusion matrix and error rate is printed
def main():

    #Import data, reshape
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train_reshaped = x_train.reshape(len(x_train), len(x_train[0])**2)/255
    x_test_reshaped = x_test.reshape(len(x_test), len(x_test[0])**2)/255
    print(len(x_train_reshaped[0]))


    # Log-likelihood and number of gaussian-plots.
    #plotNumberLoglik(x_data=x_train_reshaped, y_data=y_train, number=5, dimensions=20,
      #               c=10, iterationsCMeans=100, iterationsEM=200)

    #Train model

    # Dimensions to run the prediction, to compare test accuracy.
    dimensions = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    errors = []

    # The digit 1 only has 1 cluster, and will give error if it is initialized with more
    # for higher dimensions
    c=[10,1,10,10,10,10,10,10,10,10]
    #c=[2, 1, 6, 8, 8, 2, 2, 3, 1, 2]

    # For each number of dimensions in PCA
    for dim in dimensions:

        print("--- Running with dim = "+str(dim)+" ---")

        # The digits that become only a single gaussian with higher dimensions sometimes
        # causes errors if they are initialized with more than a single gaussian. They are
        # therefore set to 1 when they would give error.
        if dim==30:
            c[7]=1
            c[8]=1

        # At this point all digits only have a single gaussian, and initializing with more
        # sometimes gives error
        if dim==50:
            c=[1 for i in range(10)]

        # train the model. c is a list containing the number of clusters for each digit, found after analysing
        # result from line 171 and 172.
        piList, myList, sigmaList, pcaList = trainModel(x_train=x_train_reshaped, y_train=y_train,
                                                        lambd=0.2, dimensions=dim, c=c,
                                                        iterationsCMeans=100, iterationsEM=200)

        print(piList)

        # Perform classification
        y_predict = predict(x_test_reshaped, piList, myList, sigmaList, pcaList)


        # Make confusion matrix and calculate error rate
        conf = confusion_matrix(y_test, y_predict)

        print(conf)
        print(1-(np.sum(np.diag(conf))/np.sum(conf)))
        errors.append(1-(np.sum(np.diag(conf))/np.sum(conf)))


    # Plot error as a function of dimensions in PCA
    plt.figure()
    plt.plot(dimensions, errors)
    plt.xlabel("dimensions in PCA")
    plt.ylabel("Test error")
    plt.show()