import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# Draws a random sample from a gaussian mixture model
def predict(d, pis, mys, sigmas):

    # Random number to determine which gaussian to draw from
    randomNumber=np.random.uniform()

    # Cumulative of pis
    current=0

    for i in range(len(pis)):

        # Add pi_i to current
        current += pis[i]

        # If current is higher than the random number, then we draw from the belonging gaussian
        if current>randomNumber:
            x=np.random.multivariate_normal(mys[i],sigmas[i])
            return x

    return


#Draws n samples from gaussian mixture model
def drawTrainSet(n, d, pis, mys, sigmas):

    train=np.ndarray((n,d))

    # Calls predict-function n times
    for i in range(n):
        train[i]=predict(d, pis, mys, sigmas)

    return train


#Draws n samples from gaussian mixture model, but keeps them seperated in list
def drawSeveralTrainSet(n, dim, pis, mys, sigmas):

    trainSets = []

    # Calls drawTrainSet for each gaussian mixture model, and add to trainSets-variable
    for i in range(len(pis)):
        trainSets.append(drawTrainSet(int(n*pis[i]), dim, [1], [mys[i]], [sigmas[i]]))

    return trainSets


# Merges a seperated list of training data into a single data structure
def mergeTrainSets(trainSets, dim):

    # Makes an line of zero, to be able to concatenate
    totalTrain = [np.zeros(dim)]

    # Concatenates the separated training data
    for i in trainSets:
        totalTrain = np.concatenate((totalTrain, i), axis=0)

    # Deletes the first row of zeros
    totalTrain = np.delete(totalTrain, 0, axis=0)

    return totalTrain


# Returns the true distribution of a gaussian mixture model as a function of x
def trueDist(x, probs, mys, sigmas):

    dist=0

    #Add each gaussian of the gaussian mixture model
    for i in range(len(probs)):
        dist+= probs[i]*scipy.stats.multivariate_normal(mys[i], sigmas[i]).pdf(x)

    return dist

# Runs the EM-algorithm to estimate probabilities, means and covariance for gaussian mixture model
def EMAlgorithm(train, pis, mys, sigmas, iter, x=None, unknown_mys=False,
                unknown_sigmas=False, plot=False, lambd=0):

    # Lists to store log-likelihood and number of gaussians as function of iterations
    logLikList = np.zeros(iter)
    numberGaussiansList=np.zeros(iter)

    #If plot: Plot model initial model
    if plot:
        y = trueDist(x, pis, mys, sigmas)
        plt.plot(x, y, '--', label='Iteration 0')

    z=np.zeros((len(pis),len(train)))

    # For each iteration
    for i in range(iter):

        #print("--- Iteration "+str(i+1)+" ---")

        #Calculates z as in formula on page 53 in class slides
        for j in range(len(pis)):
            z[j] = pis[j] * scipy.stats.multivariate_normal(mys[j], sigmas[j]).pdf(train)
        newz = z/np.sum(z, axis=0)

        # This loop controls error that arises. If z contains values that are too small, python will
        # have presicion error, resulting in negative infinite values. This will result in NaN, and error.
        # This mechanism removes the gaussians that have too small z, thus removing the problem.
        # This while loop thus removes gaussians that are not probable, thus allowing the algorithm to
        # remove gaussians from the model.
        #while np.min(newz)<1e-30:
        while np.log(np.min(newz))==-np.inf:

            #index of gaussian to remove
            index=np.argmin(newz)//len(newz[0])

            pis = np.delete(pis, index)
            mys = np.delete(mys, index, axis=0)
            sigmas = np.delete(sigmas, index, axis=0)
            z = np.delete(z, index, axis=0)
            newz = z/np.sum(z, axis=0)

        # Calculates log-likelihood as in page 52 in class slides, and adds numbers of remaining gaussians.
        logLikList[i] = np.sum(np.log(np.sum(z, axis=0)))
        #print("Log-lik: ", logLikList[i])
        numberGaussiansList[i] = len(pis)

        # Final z after removing gaussians that give error
        z=newz

        # Numerator from equation (74) in page 59 in slides
        Nk = np.sum(z * (1 + lambd * np.log(z)), axis=1)

        # Calculates new pis as in page 59 in slides. lambd=0 gives pis as in page 53 in slides
        pis = Nk / np.sum(Nk)
        #print("Pis: ", pis)

        # Calculates mys as in page 59 in slides
        if unknown_mys:
            mys = np.dot(z * (1 + lambd * np.log(z)), train) / Nk[:, np.newaxis]
            #print("Mys: ", mys)

        # Calculates sigmas as in page 59 in slides
        if unknown_sigmas:
            for j in range(len(pis)):
                sigmas[j] = np.dot(np.transpose(train-mys[j])*z[j] * (1 + lambd * np.log(z[j])),train-mys[j])/Nk[j]
            #print("Sigmas: ", sigmas)

        # If plot: Plots model after each iteration
        if plot:
            y=trueDist(x, pis, mys, sigmas)
            plt.plot(x,y,'--', label='Iteration '+str(i+1))

    return pis, mys, sigmas, logLikList, numberGaussiansList


# Performs the C-means algorithm on training data
def cMeans(train, c, dim, rangeMin, rangeMax, iterations):

    # Random initial centers
    centers=np.random.uniform(rangeMin, rangeMax, size=(c,dim))
    initialU=np.zeros((c, len(train)))

    #Parameter to measure ||newU-oldU||, as stopping criterion
    diff=1

    # For each iteration step, until ||oldU-newU|| is smaller than threshold
    while diff>1e-3:

        oldU = initialU.copy()
        # Calculates U based on current centers (see report for formula)
        for j in range(c):
            initialU[j]=np.sum((train-centers[j])**2, axis=1)
        initialU = 1 / (initialU * np.sum(1 / initialU, axis=0))

        # Set diff to ||oldU-newU||, to be used as stopping criterion
        diff = np.linalg.norm(oldU-initialU, ord=2)

        # Calculates new centers based on U (see report for formula)
        centers = np.matmul(initialU ** 2, train) / (np.sum(initialU**2, axis=1)).reshape(c, 1)

    # Estimates pis based on U
    pis = np.sum(initialU, axis=1)/len(train)

    # Estimates sigmas based on U
    sigmas = np.zeros(((c, dim, dim)))
    numberList = np.zeros(c)
    uTranspose = np.transpose(initialU)

    #For each training sample
    for i in range(len(train)):

        # Choose most probable center, calculate (x-mu)(x-mu)^T
        index = np.argmax(uTranspose[i])
        distance = np.asarray(train[i] - centers[index])
        numberList[index] += 1
        sigmas[index] += np.outer(distance, distance)

    # Divide by numbers belonging to each center
    for i in range(c):
        sigmas[i] /= numberList[i]


    return centers, pis, sigmas

