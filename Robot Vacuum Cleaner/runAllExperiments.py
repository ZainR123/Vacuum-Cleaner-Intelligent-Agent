import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway

from simpleBot import *


# OWN CODE START ===============================================================================================
# Run each bot amount 10 times
# Calculate average dirt collection, map coverage and collisions
# Optional heatmap can be generated
def runSetOfExperiments(numberOfRuns, numberOfBots, algorithm, collisionAlg):
    visitedGridList = []
    collisionsList = []
    visited = np.zeros((20, 20), dtype=np.int16)

    avgCoverage = 0
    avgDirtCollected = 0
    avgCollisions = 0

    for _ in range(numberOfRuns):
        dirtCollected, visitedGrid, collisions = main(numberOfBots, algorithm, collisionAlg)
        noOfZeros = np.count_nonzero(visitedGrid == 0)

        avgCoverage += ((400 - noOfZeros) / 400) * 100
        avgCollisions += collisions
        avgDirtCollected += dirtCollected

        collisionsList.append(collisions)
        visitedGridList.append(visitedGrid)

    avgCoverage /= numberOfRuns
    avgDirtCollected /= numberOfRuns
    avgCollisions /= numberOfRuns

    if len(visitedGridList) == 1:
        visited = visitedGridList[0]

    for x in range(len(visitedGridList) - 1):

        if x == 0:
            visited = [[visitedGridList[x][i][j] + visitedGridList[x + 1][i][j] for j in range(len(visitedGridList[x][0]))] for i in range(len(visitedGridList[x]))]

        else:
            visited = [[visited[i][j] + visitedGridList[x + 1][i][j] for j in range(len(visited[0]))] for i in range(len(visited))]

    # Print Heat Map
    # if collisionAlg is None:
    #     plt.imshow(visited, cmap='hot', interpolation='nearest')
    #     title = "Heatmap of " + str(numberOfBots) + " bot coverage:" + algorithm
    #     plt.title(title)
    #     plt.show()

    print("-------------------------------------------------------------------------------------------")
    print("Bot", numberOfBots, ":", numberOfRuns, "Runs - Average Collisions: ", avgCollisions)
    print("Bot", numberOfBots, ":", numberOfRuns, "Runs - Average Coverage: ", avgCoverage)
    print("Bot", numberOfBots, ":", numberOfRuns, "Runs - Average Dirt Collected: ", avgDirtCollected)
    print("-------------------------------------------------------------------------------------------")
    return avgDirtCollected, avgCoverage, avgCollisions, collisionsList


# Run every movement algorithm and collision algorithm and print relevant graphs and data (line, boxplot and statistics)
def runExperimentsWithDifferentParameters():
    algorithms = ["random", "spiral", "cell search", "cell search dirt", "attract", "repel"]

    collisionAlgs = [None, "dangerThreshold", "logisticRegression"]

    collisionAlgsNames = ["distance-based (control)", "danger threshold", "logistic regression"]
    numberOfRuns = 10
    nOfBots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    collisionsPerCollisionAlgorithm = []
    averageCollisionsPerCollisionAlgorithm = []
    averageCoveragePerCollisionAlgorithm = []
    averageDirtPerCollisionAlgorithm = []
    finalResultsList = []
    finalAvgDirt = []
    finalAvgCoverage = []
    finalAvgCollisions = []

    for j in collisionAlgs:
        print("===========================================================================================")
        print("-------------------------------------------------------------------------------------------")
        print("===========================================================================================")
        print(j)
        print("===========================================================================================")
        print("-------------------------------------------------------------------------------------------")
        print("===========================================================================================")

        summedAverageCollisionList = []
        summedAverageCoverageList = []
        summedAverageDirtList = []
        allAlgorithmsCollisionsList = []
        resultsList = []

        for i in algorithms:
            averageDirtCollectedList = []
            averageCoverageList = []
            averageCollisionsList = []

            print("===========================================================================================")
            print(i)
            print("===========================================================================================")

            for numberOfBots in range(1, (len(nOfBots) + 1)):
                avgDirtCollected, avgCoverage, avgCollisions, collisionsList = runSetOfExperiments(numberOfRuns, numberOfBots, i, j)

                averageDirtCollectedList.append(avgDirtCollected)
                averageCoverageList.append(avgCoverage)
                averageCollisionsList.append(avgCollisions)

                allAlgorithmsCollisionsList.extend(collisionsList)

            resultsList.append((averageDirtCollectedList, averageCoverageList, averageCollisionsList))

        for x in range(len(resultsList) - 1):

            if x == 0:
                summedAverageCollisionList = [a + b for a, b in zip(resultsList[0][2], resultsList[1][2])]
                summedAverageCoverageList = [a + b for a, b in zip(resultsList[0][1], resultsList[1][1])]
                summedAverageDirtList = [a + b for a, b in zip(resultsList[0][0], resultsList[1][0])]
            else:
                summedAverageCollisionList = [a + b for a, b in zip(summedAverageCollisionList, resultsList[x + 1][2])]
                summedAverageCoverageList = [a + b for a, b in zip(summedAverageCoverageList, resultsList[x + 1][1])]
                summedAverageDirtList = [a + b for a, b in zip(summedAverageDirtList, resultsList[x + 1][0])]

        summedAverageCollisionList = [x / 6 for x in summedAverageCollisionList]
        summedAverageCoverageList = [x / 6 for x in summedAverageCoverageList]
        summedAverageDirtList = [x / 6 for x in summedAverageDirtList]

        finalResultsList.append((summedAverageDirtList, summedAverageCoverageList, summedAverageCollisionList))

        averageCollisionsPerCollisionAlgorithm.append(summedAverageCollisionList)
        averageCoveragePerCollisionAlgorithm.append(summedAverageCoverageList)
        averageDirtPerCollisionAlgorithm.append(summedAverageDirtList)

        collisionsPerCollisionAlgorithm.append(allAlgorithmsCollisionsList)

        print("================================================================================================================")
        print(algorithms, "RESULTS")
        print(resultsList)
        print("================================================================================================================")

        plt.title('number of bots vs average dirt collected')
        plt.plot(nOfBots, resultsList[0][0], label="random")
        plt.plot(nOfBots, resultsList[1][0], label="spiral")
        plt.plot(nOfBots, resultsList[2][0], label="cell search")
        plt.plot(nOfBots, resultsList[3][0], label="cell search dirt")
        plt.plot(nOfBots, resultsList[4][0], label="attract")
        plt.plot(nOfBots, resultsList[5][0], label="repel")
        plt.xlabel('number of bots')
        plt.ylabel('average dirt collected')
        plt.xticks(np.arange(1, len(nOfBots) + 1, 1))
        plt.legend()
        plt.show()

        plt.title('number of bots vs average map coverage')
        plt.plot(nOfBots, resultsList[0][1], label="random")
        plt.plot(nOfBots, resultsList[1][1], label="spiral")
        plt.plot(nOfBots, resultsList[2][1], label="cell search")
        plt.plot(nOfBots, resultsList[3][1], label="cell search dirt")
        plt.plot(nOfBots, resultsList[4][1], label="attract")
        plt.plot(nOfBots, resultsList[5][1], label="repel")
        plt.xlabel('number of bots')
        plt.ylabel('average map coverage')
        plt.xticks(np.arange(1, len(nOfBots) + 1, 1))
        plt.legend()
        plt.show()

        plt.title('number of bots vs average bot collisions')
        plt.plot(nOfBots, resultsList[0][2], label="random")
        plt.plot(nOfBots, resultsList[1][2], label="spiral")
        plt.plot(nOfBots, resultsList[2][2], label="cell search")
        plt.plot(nOfBots, resultsList[3][2], label="cell search dirt")
        plt.plot(nOfBots, resultsList[4][2], label="attract")
        plt.plot(nOfBots, resultsList[5][2], label="repel")
        plt.xlabel('number of bots')
        plt.ylabel('average bot collisions')
        plt.xticks(np.arange(1, len(nOfBots) + 1, 1))
        plt.legend()
        plt.show()

    print("================================================================================================================")
    print("finalResultsList")
    print(finalResultsList)
    print("================================================================================================================")

    for x in range(len(finalResultsList) - 1):

        if x == 0:
            finalAvgCollisions = [a + b for a, b in zip(finalResultsList[0][2], finalResultsList[1][2])]
            finalAvgCoverage = [a + b for a, b in zip(finalResultsList[0][1], finalResultsList[1][1])]
            finalAvgDirt = [a + b for a, b in zip(finalResultsList[0][0], finalResultsList[1][0])]
        else:
            finalAvgCollisions = [a + b for a, b in zip(finalAvgCollisions, finalResultsList[x + 1][2])]
            finalAvgCoverage = [a + b for a, b in zip(finalAvgCoverage, finalResultsList[x + 1][1])]
            finalAvgDirt = [a + b for a, b in zip(finalAvgDirt, finalResultsList[x + 1][0])]

    finalAvgDirt = [x / 3 for x in finalAvgDirt]
    finalAvgCoverage = [x / 3 for x in finalAvgCoverage]
    finalAvgCollisions = [x / 3 for x in finalAvgCollisions]

    print("================================================================================================================")
    print("Average Collisions per collision algorithm")
    print(averageCollisionsPerCollisionAlgorithm)
    print("================================================================================================================")

    plt.title('number of bots vs average collisions')
    plt.plot(nOfBots, averageCollisionsPerCollisionAlgorithm[0], label="distance-based")
    plt.plot(nOfBots, averageCollisionsPerCollisionAlgorithm[1], label="danger threshold")
    plt.plot(nOfBots, averageCollisionsPerCollisionAlgorithm[2], label="logistic regression")
    plt.xlabel('number of bots')
    plt.ylabel('average collisions')
    plt.xticks(np.arange(1, len(nOfBots) + 1, 1))
    plt.legend()
    plt.show()

    print(ttest_ind(collisionsPerCollisionAlgorithm[1], collisionsPerCollisionAlgorithm[2]))
    print(f_oneway(collisionsPerCollisionAlgorithm[0], collisionsPerCollisionAlgorithm[1], collisionsPerCollisionAlgorithm[2]))

    print("================================================================================================================")
    print("Collisions per collision algorithm")
    print(collisionsPerCollisionAlgorithm)
    print("================================================================================================================")

    plt.title('boxplot of collisions vs collision algorithms')
    plt.boxplot(collisionsPerCollisionAlgorithm, labels=collisionAlgsNames, showfliers=False)
    plt.xlabel('collision algorithms')
    plt.ylabel('collisions over all algorithms')
    plt.show()

    collisionsPerCollisionAlgorithm[0] = average(collisionsPerCollisionAlgorithm[0])
    collisionsPerCollisionAlgorithm[1] = average(collisionsPerCollisionAlgorithm[1])
    collisionsPerCollisionAlgorithm[2] = average(collisionsPerCollisionAlgorithm[2])

    plt.title('collision algorithms vs average collisions')
    plt.bar(collisionAlgsNames, collisionsPerCollisionAlgorithm, color='maroon')
    plt.xlabel('collision algorithms')
    plt.ylabel('average collisions over all algorithms')
    plt.show()

    print("================================================================================================================")
    print("finalAvgCoverage")
    print(finalAvgCoverage)
    print("================================================================================================================")

    plt.title('number of bots vs average coverage over all algorithms')
    plt.plot(nOfBots, finalAvgCoverage)
    plt.xlabel('number of bots')
    plt.ylabel('average coverage')
    plt.xticks(np.arange(1, len(nOfBots) + 1, 1))
    plt.legend()
    plt.show()

    print("================================================================================================================")
    print("finalAvgDirt")
    print(finalAvgDirt)
    print("================================================================================================================")

    plt.title('number of bots vs average dirt collected over all algorithms')
    plt.plot(nOfBots, finalAvgDirt)
    plt.xlabel('number of bots')
    plt.ylabel('average dirt collected')
    plt.xticks(np.arange(1, len(nOfBots) + 1, 1))
    plt.legend()
    plt.show()

    print("================================================================================================================")
    print("finalAvgCollisions")
    print(finalAvgCollisions)
    print("================================================================================================================")

    plt.title('number of bots vs average collisions over all algorithms')
    plt.plot(nOfBots, finalAvgCollisions)
    plt.xlabel('number of bots')
    plt.ylabel('average collisions')
    plt.xticks(np.arange(1, len(nOfBots) + 1, 1))
    plt.legend()
    plt.show()


def average(lst):
    return sum(lst) / len(lst)


runExperimentsWithDifferentParameters()

# OWN CODE END===============================================================================================
