import functions as fc
import plotting as pt

def task1():

    #Parameters
    truePis = [0.5, 0.3, 0.2]
    trueMys = [[0], [2], [5]]
    trueSigmas = [[[1]], [[1]], [[1]]]
    initialPis = [0.8, 0.1, 0.1]


    # Plots the true distribution with the parameters above and in argument.
    # Draws training data, and plots it as histogram.
    # If plotMix=True: Plots the different gaussian models of the true distribution.
    # If plotAlgorithm=True: Runs the EM-algorithm on the training data, and plots
    # the trained model after each iteration.
    pt.plotEM(truePis=truePis, trueMys=trueMys, trueSigmas=trueSigmas, initialPis=initialPis,
              initialMys=trueMys, initialSigmas=trueSigmas, dim=1, plotRange=[-5, 10],
              nTrain = 10000, iter=20, unknown_mys=False, unknown_sigmas=False, lambd=0,
              plotMix=False, plotAlgorithm=True)


def task2():

    # Parameters
    truePis = [0.7, 0.3]
    trueMys = [[2], [5]]
    trueSigmas = [[[1]], [[1]]]
    initialPis = [0.1, 0.9]
    initialMys = [[1], [2]]

    # Plots the true distribution with the parameters above and in argument.
    # Draws training data, and plots it as histogram.
    # If plotMix=True: Plots the different gaussian models of the true distribution.
    # If plotAlgorithm=True: Runs the EM-algorithm on the training data, and plots
    # the trained model after each iteration.
    pt.plotEM(truePis=truePis, trueMys=trueMys, trueSigmas=trueSigmas, initialPis=initialPis,
              initialMys=initialMys, initialSigmas=trueSigmas, dim=1, plotRange=[-5, 10],
              nTrain=10000, iter=15, unknown_mys=True, unknown_sigmas=False, lambd=0,
              plotMix=False, plotAlgorithm=True)


def task3():

    #Parameters
    truePis = [0.7, 0.3]
    trueMys = [[2], [5]]
    trueSigmas = [[[3]], [[0.4]]]
    initialPis=[0.1, 0.9]
    initialSigmas=[[[1]],[[1]]]

    # Plots the true distribution with the parameters above and in argument.
    # Draws training data, and plots it as histogram.
    # If plotMix=True: Plots the different gaussian models of the true distribution.
    # If plotAlgorithm=True: Runs the EM-algorithm on the training data, and plots
    # the trained model after each iteration.
    pt.plotEM(truePis=truePis, trueMys=trueMys, trueSigmas=trueSigmas, initialPis=initialPis,
              initialMys=trueMys, initialSigmas=initialSigmas, dim=1, plotRange=[-5, 10],
              nTrain=10000, iter=10, unknown_mys=False, unknown_sigmas=True, lambd=0,
              plotMix=False, plotAlgorithm=True)


def task4():

    #Four different examples are run in the report, to analyse the behaviour.
    run=4 # this works with removing criterion 1e-30

    #First example
    if run == 1 or run == 2:
        truePis = [0.7, 0.3]
        trueMys = [[2], [5]]
        trueSigmas = [[[1]], [[1]]]

    if run == 1:
        initialPis = [[0.1], [0.9]]
        initialMys = [[1], [2]]
        initialSigmas = [[[1]], [[1]]]

    #Second example
    if run == 2: #20 iterations to remove one gaussian
        initialPis = [0.3, 0.3, 0.4]
        initialMys = [[1], [1.5], [2]]
        initialSigmas = [[[1]], [[1]], [[1]]]

    #Third example
    if run == 3 or run == 4:
        truePis = [0.2, 0.5, 0.3]
        trueMys = [[1], [4], [8]]
        trueSigmas = [[[1]], [[1]], [[1]]]
        initialPis = [0.3, 0.3, 0.4]
        initialMys = [[1], [3], [-5]]
        initialSigmas = [[[1]], [[1]], [[1]]]

    #Fourth example
    if run == 4:
        initialMys = [[1], [3], [5]]


    # Plots the true distribution with the parameters above and in argument.
    # Draws training data, and plots it as histogram.
    # If plotMix=True: Plots the different gaussian models of the true distribution.
    # If plotAlgorithm=True: Runs the EM-algorithm on the training data, and plots
    # the trained model after each iteration.
    pt.plotEM(truePis=truePis, trueMys=trueMys, trueSigmas=trueSigmas, initialPis=initialPis,
              initialMys=initialMys, initialSigmas=initialSigmas, dim=1, plotRange=[-5, 10],
              nTrain=10000, iter=20, unknown_mys=True, unknown_sigmas=False, lambd=0,
              plotMix=False, plotAlgorithm=True)


def task5():

    #Parameters
    truePis = [0.05, 0.09, 0.07, 0.11, 0.08, 0.19, 0.14, 0.12, 0.07, 0.08]
    trueMys = [[1, 1], [1, 20], [6, 10], [10, 16], [12, 24], [20, 22], [18, 2], [18, 5], [12, 20], [6, 7]]
    trueSigmas = [[[1, 0.7], [0.7, 2]], [[1, -0.6], [-0.6, 5]], [[3, -0.2], [-0.2, 0.5]], [[2, 0], [0, 2]],
                  [[3, 0.3], [0.3, 1]], [[1, 0.9], [0.9, 4]], [[2, 0.1], [0.1, 1.5]], [[6, 2], [2, 3]],
                  [[1, 0.1], [0.1, 2]], [[1, -0.2], [-0.2, 3]]]


    # Draws the training data from the true distribution with parameters as above, and plots the
    # training data as a scatter plot, together with 95% confidence ellipses of each gaussian.
    # Runs the C-means clustering algorithm.
    # If plotCenters=True: Plot the resulting means from the C-means algorithm.
    # If plotInitial=True: Plots the 95% confidence ellipses for the empirical means and
    # covariance matrices from the C-means algorithm.
    # If plotFinal=True: Runs the EM-algorithm and plots 95% confidence ellipses from the fitted model.
    # If plotLoglik=True: Plots the log-likelihood and number of gaussians as a
    # function of iteration number.
    pt.plotEM2D(truePis=truePis, trueMys=trueMys, trueSigmas=trueSigmas,
                lambd = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
                numCenters = 15, iterationsCMeans = 100, iterationsEM = 200,
                plotCenters = False, plotInitial=False, plotFinal=True, plotLoglik=True)


