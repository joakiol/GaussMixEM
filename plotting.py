import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import functions as fc

#Plots each gaussian for gaussian mixture model as a function of x
def plot_mixtures(pis, mys, sigmas, x):

    y=np.zeros((len(pis), len(x)))

    # For each gaussian, plot y=p(x)
    for i in range(len(pis)):
        y = pis[i] * scipy.stats.multivariate_normal(mys[i], sigmas[i]).pdf(x)
        plt.plot(x,y, '--', label='Mixture '+str(i+1))


# Plots 95% confidence ellipses for gaussian mixture model
def plot95ConfEllipse(pis, mys, sigmas, ax, color):

    # For each gaussian
    for i in range(len(pis)):

        # Calculate half axis lengths and angle of ellipse
        eigenval, eigenvec=np.linalg.eig(sigmas[i])
        ellipseAxis=2*np.sqrt(5.991*eigenval)
        if eigenvec[0][1]==0:
            ellipseAngle=np.pi/2
        else:
            ellipseAngle=np.arctan(eigenvec[0][0]/eigenvec[0][1])

        # Plots ellipse
        ell = Ellipse(xy=(mys[i][0], mys[i][1]),
                  width=ellipseAxis[0], height=ellipseAxis[1],
                  angle=90+np.rad2deg(ellipseAngle), edgecolor=color, facecolor='none', linewidth=2, alpha=1)
        ax.add_artist(ell)

    return ax






# Performs the tasks necessary to obtain the corresponding plots for 1D-tasks
def plotEM(truePis, trueMys, trueSigmas, initialPis, initialMys, initialSigmas, dim, plotRange,
           nTrain, iter, unknown_mys, unknown_sigmas, lambd, plotMix, plotAlgorithm):

    # Plots the true distribution with the parameters as in argument
    x = np.linspace(plotRange[0], plotRange[1], 200)
    y = fc.trueDist(x, truePis, trueMys, trueSigmas)
    plt.figure()
    plt.plot(x, y, label='True dist', color='b')
    plt.xlabel("x")
    plt.ylabel("p(x)")

    # Draws training data, and plots it as histogram.
    train = fc.drawTrainSet(nTrain, dim, truePis, trueMys, trueSigmas)
    plt.hist(train, 30, normed=1, rwidth=0.6, alpha=0.5, color='y')

    # If plotMix=True: Plots the different gaussian models of the true distribution.
    if plotMix:
        plot_mixtures(truePis, trueMys, trueSigmas, x)

    # If plotAlgorithm=True: Runs the EM-algorithm on the training data, and plots
    # the trained model after each iteration.
    if plotAlgorithm:
        fc.EMAlgorithm(train, initialPis, initialMys, initialSigmas, iter, unknown_mys = unknown_mys,
                       unknown_sigmas = unknown_sigmas, x=x, plot=True, lambd=lambd)

    plt.legend()
    plt.show()


# Plots training data as scatter plot with 95% confidence ellipses for each gaussian in mixture model
def plotTrainSets(trainSets, pis, mys, sigmas, ax):

    # Colors for each gaussian
    colors = ['red', 'green', 'blue', 'sandybrown', 'orange', 'c', 'magenta', 'darkgreen', 'blueviolet', 'pink']

    # For each gaussian, scatter plot training data belonging to given gaussian, and
    # plot 95% confidence ellipse
    for i in range(len(trainSets)):
        transposed = np.transpose(trainSets[i])
        plt.scatter(transposed[0], transposed[1], marker='.', color = colors[i])
        ax = plot95ConfEllipse([pis[i]], [mys[i]], [sigmas[i]], ax, color = colors[i])

    return ax


# Performs the tasks necessary to obtain the corresponding plots for 2D-task
def plotEM2D(truePis, trueMys, trueSigmas, lambd, numCenters, iterationsCMeans,
             iterationsEM, plotCenters, plotInitial, plotFinal, plotLoglik):

    # Draws the training data from the true distribution with parameters as above, and plots the
    # training data as a scatter plot, together with 95% confidence ellipses of each gaussian.
    ax = plt.subplot(111)
    trainSets = fc.drawSeveralTrainSet(n=1000, dim=2, pis=truePis, mys=trueMys, sigmas=trueSigmas)
    train = fc.mergeTrainSets(trainSets, dim=2)

    # If plotFinal=True there will be subplots, thus this will not be performed
    if not plotFinal:
        ax = plotTrainSets(trainSets, truePis, trueMys, trueSigmas, ax)

    # Runs the C-means clustering algorithm.
    initialMys, initialPis, initialSigmas = fc.cMeans(train, numCenters, dim=2, rangeMin=0, rangeMax=20, iterations = iterationsCMeans)

    # If plotCenters=True: Plot the resulting means from the C-means algorithm.
    if plotCenters and not plotFinal:
        plt.scatter(np.transpose(initialMys)[0], np.transpose(initialMys)[1], color='black', s=100)
        plt.xlabel("x1")
        plt.ylabel("x2")

    # If plotInitial=True: Plots the 95% confidence ellipses for the empirical means and
    # covariance matrices from the C-means algorithm.
    if plotInitial and not plotFinal:
        ax = plot95ConfEllipse(initialPis, initialMys, initialSigmas, ax, color='black')

    # If plotFinal=True: Runs the EM-algorithm and plots 95% confidence ellipses from the fitted model.
    if plotFinal or plotLoglik:
        finalPis = []
        finalMys = []
        finalSigmas = []
        finalLoglik = []
        finalNumbergauss = []

        # For each lambd in lambd-list
        for i in lambd:

            # Run algorithm and add result
            newPis, newMys, newSigmas, loglik, numberGauss = fc.EMAlgorithm(train, initialPis, initialMys,
                                                                            initialSigmas, iterationsEM,
                                                                            unknown_mys=True, unknown_sigmas=True,
                                                                            lambd=i)
            finalPis.append(newPis)
            finalMys.append(newMys)
            finalSigmas.append(newSigmas)
            finalLoglik.append(loglik)
            finalNumbergauss.append(numberGauss)

    # For each lambda, subplot the training data with true and estimated 95% confidence ellipses
    if plotFinal:

        ax1 = plt.subplot(241)
        ax1 = plotTrainSets(trainSets, truePis, trueMys, trueSigmas, ax1)
        ax1 = plot95ConfEllipse(finalPis[0], finalMys[0], finalSigmas[0], ax1, color='black')
        plt.title("Lambda = "+str(lambd[0]))
        plt.xlabel("x1")
        plt.xlabel("x2")

        ax2 = plt.subplot(242)
        ax2 = plotTrainSets(trainSets, truePis, trueMys, trueSigmas, ax2)
        ax2 = plot95ConfEllipse(finalPis[1], finalMys[1], finalSigmas[1], ax2, color='black')
        plt.title("Lambda = " + str(lambd[1]))
        plt.xlabel("x1")
        plt.xlabel("x2")

        ax3 = plt.subplot(243)
        ax3 = plotTrainSets(trainSets, truePis, trueMys, trueSigmas, ax3)
        ax3 = plot95ConfEllipse(finalPis[2], finalMys[2], finalSigmas[2], ax3, color='black')
        plt.title("Lambda = " + str(lambd[2]))
        plt.xlabel("x1")
        plt.xlabel("x2")

        ax4 = plt.subplot(244)
        ax4 = plotTrainSets(trainSets, truePis, trueMys, trueSigmas, ax4)
        ax4 = plot95ConfEllipse(finalPis[3], finalMys[3], finalSigmas[3], ax4, color='black')
        plt.title("Lambda = " + str(lambd[3]))
        plt.xlabel("x1")
        plt.xlabel("x2")

        ax5 = plt.subplot(245)
        ax5 = plotTrainSets(trainSets, truePis, trueMys, trueSigmas, ax5)
        ax5 = plot95ConfEllipse(finalPis[4], finalMys[4], finalSigmas[4], ax5, color='black')
        plt.title("Lambda = " + str(lambd[4]))
        plt.xlabel("x1")
        plt.xlabel("x2")

        ax6 = plt.subplot(246)
        ax6 = plotTrainSets(trainSets, truePis, trueMys, trueSigmas, ax6)
        ax6 = plot95ConfEllipse(finalPis[5], finalMys[5], finalSigmas[5], ax6, color='black')
        plt.title("Lambda = " + str(lambd[5]))
        plt.xlabel("x1")
        plt.xlabel("x2")

        ax7 = plt.subplot(247)
        ax7 = plotTrainSets(trainSets, truePis, trueMys, trueSigmas, ax7)
        ax7 = plot95ConfEllipse(finalPis[6], finalMys[6], finalSigmas[6], ax7, color='black')
        plt.title("Lambda = " + str(lambd[6]))
        plt.xlabel("x1")
        plt.xlabel("x2")

        ax8 = plt.subplot(248)
        ax8 = plotTrainSets(trainSets, truePis, trueMys, trueSigmas, ax8)
        ax8 = plot95ConfEllipse(finalPis[7], finalMys[7], finalSigmas[7], ax8, color='black')
        plt.title("Lambda = " + str(lambd[7]))
        plt.xlabel("x1")
        plt.xlabel("x2")


    plt.tight_layout()
    plt.show()

    # If plotLoglik=True: Plots the log-likelihood and number of gaussians as a
    # function of iteration number.
    if plotLoglik:

        iterations = np.linspace(1, iterationsEM, iterationsEM)
        plt.subplot(121)

        # For each lambd in lambd-list, plot log-likelihood
        for i in range(len(finalLoglik)):
            plt.plot(iterations, finalLoglik[i], label = "Lambda = "+str(lambd[i]))
        plt.xlabel("Iterations")
        plt.ylabel("Log-likelihood")
        plt.legend()


        plt.subplot(122)

        # For each lambd in lambd-list, plot number of gaussians
        for i in range(len(finalNumbergauss)):
            plt.plot(iterations, finalNumbergauss[i], label = "Lambda = "+str(lambd[i]))
        plt.xlabel("Iterations")
        plt.ylabel("# of mixtures")
        plt.legend()

        plt.show()