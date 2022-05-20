import math
import os
import random
import tkinter as tk
from itertools import chain

import numpy as np

from joblib import load, dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split


class Counter:
    def __init__(self):
        self.totalMoves = 0
        self.dirtCollected = 0
        self.totalCollisions = 0
        self.visitedGrid = np.zeros((20, 20), dtype=np.int16)


class Charger:
    def __init__(self, namep):
        self.centreX = 400
        self.centreY = 200
        self.name = namep

    def draw(self, canvas):
        canvas.create_oval(self.centreX - 10, self.centreY - 10,
                           self.centreX + 10, self.centreY + 10,
                           fill="gold", tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY


class Bin:
    def __init__(self, namep):
        self.centreX = 700
        self.centreY = 800
        self.name = namep

    def draw(self, canvas):
        canvas.create_oval(self.centreX - 10, self.centreY - 10,
                           self.centreX + 10, self.centreY + 10,
                           fill="purple", tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY


class Dirt:
    def __init__(self, namep, xx, yy):
        self.centreX = xx
        self.centreY = yy
        self.name = namep

    def draw(self, canvas):
        canvas.create_oval(self.centreX - 1, self.centreY - 1,
                           self.centreX + 1, self.centreY + 1,
                           fill="grey", tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY


class Bot:

    # initialise bot variables
    def __init__(self, namep, position):
        self.x = position[0]
        self.y = position[1]
        self.theta = position[2]
        self.name = namep
        self.ll = 60
        self.vl = 0.0
        self.vr = 0.0
        self.maxBattery = 1000
        self.battery = self.maxBattery
        self.capacity = 200
        self.currentDirt = 0
        self.turning = 0
        self.moving = 0
        self.currentlyTurning = False
        self.view = [0] * 9
        self.collisionDetected = False
        self.collisionAvoided = False
        self.noCollision = False
        self.sensorPositions = []
        self.cameraPositions = []
        self.count = 0
        self.full = False
        self.charge = False

    # draw environment elements
    def draw(self, canvas):

        self.cameraPositions = []
        for pos in range(20, -21, -5):
            self.cameraPositions.append(((self.x + pos * math.sin(self.theta)) + 30 * math.sin((math.pi / 2.0) - self.theta),
                                         (self.y - pos * math.cos(self.theta)) + 30 * math.cos((math.pi / 2.0) - self.theta)))

        points = [(self.x + 30 * math.sin(self.theta)) - 30 * math.sin((math.pi / 2.0) - self.theta),
                  (self.y - 30 * math.cos(self.theta)) - 30 * math.cos((math.pi / 2.0) - self.theta),
                  (self.x - 30 * math.sin(self.theta)) - 30 * math.sin((math.pi / 2.0) - self.theta),
                  (self.y + 30 * math.cos(self.theta)) - 30 * math.cos((math.pi / 2.0) - self.theta),
                  (self.x - 30 * math.sin(self.theta)) + 30 * math.sin((math.pi / 2.0) - self.theta),
                  (self.y + 30 * math.cos(self.theta)) + 30 * math.cos((math.pi / 2.0) - self.theta),
                  (self.x + 30 * math.sin(self.theta)) + 30 * math.sin((math.pi / 2.0) - self.theta),
                  (self.y - 30 * math.cos(self.theta)) + 30 * math.cos((math.pi / 2.0) - self.theta)]
        canvas.create_polygon(points, fill="blue", tags=self.name)

        self.sensorPositions = [(self.x + 20 * math.sin(self.theta)) + 30 * math.sin((math.pi / 2.0) - self.theta),
                                (self.y - 20 * math.cos(self.theta)) + 30 * math.cos((math.pi / 2.0) - self.theta),
                                (self.x - 20 * math.sin(self.theta)) + 30 * math.sin((math.pi / 2.0) - self.theta),
                                (self.y + 20 * math.cos(self.theta)) + 30 * math.cos((math.pi / 2.0) - self.theta)]

        centre1PosX = self.x
        centre1PosY = self.y
        canvas.create_oval(centre1PosX - 15, centre1PosY - 15,
                           centre1PosX + 15, centre1PosY + 15,
                           fill="gold", tags=self.name)
        canvas.create_text(self.x, self.y, text="battery: " + str(self.battery), tags=self.name, fill="green1")
        canvas.create_text(self.x, self.y - 10, text="dirt: " + str(self.currentDirt), tags=self.name, fill="green1")

        wheel1PosX = self.x - 30 * math.sin(self.theta)
        wheel1PosY = self.y + 30 * math.cos(self.theta)
        canvas.create_oval(wheel1PosX - 3, wheel1PosY - 3,
                           wheel1PosX + 3, wheel1PosY + 3,
                           fill="red", tags=self.name)

        wheel2PosX = self.x + 30 * math.sin(self.theta)
        wheel2PosY = self.y - 30 * math.cos(self.theta)
        canvas.create_oval(wheel2PosX - 3, wheel2PosY - 3,
                           wheel2PosX + 3, wheel2PosY + 3,
                           fill="green", tags=self.name)

        sensor1PosX = self.sensorPositions[0]
        sensor1PosY = self.sensorPositions[1]
        sensor2PosX = self.sensorPositions[2]
        sensor2PosY = self.sensorPositions[3]
        canvas.create_oval(sensor1PosX - 3, sensor1PosY - 3,
                           sensor1PosX + 3, sensor1PosY + 3,
                           fill="yellow", tags=self.name)
        canvas.create_oval(sensor2PosX - 3, sensor2PosY - 3,
                           sensor2PosX + 3, sensor2PosY + 3,
                           fill="yellow", tags=self.name)

        for xy in self.cameraPositions:
            canvas.create_oval(xy[0] - 2, xy[1] - 2, xy[0] + 2, xy[1] + 2, fill="purple1", tags=self.name)

    # cf. Dudek and Jenkin, Computational Principles of Mobile Robotics
    # how the bot moves
    def move(self, canvas, registryPassives, dt, grid, count):
        # OWN CODE START===============================================================================================
        if self.battery > 0:
            self.battery -= 1
        if self.battery == 0:
            self.vl = 0
            self.vr = 0
        for rr in registryPassives:
            if isinstance(rr, Charger) and self.distanceTo(rr) < 80 and self.charge:
                self.battery += 10

            if isinstance(rr, Bin) and self.distanceTo(rr) < 80 and self.full:
                self.currentDirt -= 10

        if self.battery >= self.maxBattery:
            self.battery = self.maxBattery
            self.charge = False

        if self.currentDirt < 0:
            self.currentDirt = 0
            self.full = False

        # OWN CODE END===============================================================================================

        if self.vl == self.vr:
            R = 0
        else:
            R = (self.ll / 2.0) * ((self.vr + self.vl) / (self.vl - self.vr))
        omega = (self.vl - self.vr) / self.ll
        ICCx = self.x - R * math.sin(self.theta)  # instantaneous centre of curvature
        ICCy = self.y + R * math.cos(self.theta)
        m = np.matrix([[math.cos(omega * dt), -math.sin(omega * dt), 0],
                       [math.sin(omega * dt), math.cos(omega * dt), 0],
                       [0, 0, 1]])
        v1 = np.matrix([[self.x - ICCx], [self.y - ICCy], [self.theta]])
        v2 = np.matrix([[ICCx], [ICCy], [omega * dt]])
        newv = np.add(np.dot(m, v1), v2)
        newX = newv.item(0)
        newY = newv.item(1)
        newTheta = newv.item(2)
        newTheta = newTheta % (2.0 * math.pi)  # make sure angle doesn't go outside [0.0,2*pi)
        self.x = newX
        self.y = newY
        self.theta = newTheta
        if self.vl == self.vr:  # straight line movement
            self.x += self.vr * math.cos(self.theta)  # vr wlog
            self.y += self.vr * math.sin(self.theta)

        # OWN CODE START===============================================================================================
        if not self.noCollision:

            if self.x <= 0.0:
                self.x = 0.0
                self.collisionDetected = True
            if self.x >= 1000.0:
                self.x = 999.0
                self.collisionDetected = True
            if self.y <= 0.0:
                self.y = 0.0
                self.collisionDetected = True
            if self.y >= 1000.0:
                self.y = 999.0
                self.collisionDetected = True

        else:

            if self.x <= 0.0:
                self.x = 0.0
                self.theta -= 1

            if self.x >= 1000.0:
                self.x = 999.0

                self.theta -= 1

            if self.y <= 0.0:
                self.y = 0.0
                self.theta -= 1

            if self.y >= 1000.0:
                self.y = 999.0
                self.theta -= 1

        if grid is not None:
            self.updateMap(canvas, grid, count)

        # OWN CODE END===============================================================================================

        canvas.delete(self.name)
        self.draw(canvas)

    # calculate if objects are in front of the bot's sensors and if so calculate the scaled distance to the bot's location
    def look(self, registryActives):
        self.view = [0] * 9
        for idx, pos in enumerate(self.cameraPositions):
            for cc in registryActives:
                # OWN CODE START===============================================================================================
                dd = self.distanceTo(cc)

                if dd != 0 and isinstance(cc, Bot):

                    if distanceToPoint(self.cameraPositions[4][0], self.cameraPositions[4][1], cc.x, cc.y) < dd:

                        # OWN CODE END===============================================================================================

                        scaledDistance = max(400 - dd, 0) / 400

                        ncx = cc.x - pos[0]
                        ncy = cc.y - pos[1]

                        m = math.tan(self.theta)
                        A = m * m + 1
                        B = 2 * (-m * ncy - ncx)
                        r = 15  # radius
                        C = ncy * ncy - r * r + ncx * ncx
                        if B * B - 4 * A * C >= 0 and scaledDistance > self.view[idx]:
                            self.view[idx] = scaledDistance

        return self.view

    # update map when a cell has been visited removing it from the grid
    def updateMap(self, canvas, grid, count):
        # OWN CODE START ===============================================================================================
        xMapPosition = int(math.floor(self.x / 50))
        yMapPosition = int(math.floor(self.y / 50))

        count.visitedGrid[xMapPosition][yMapPosition] += 1

        if self.currentDirt < self.capacity:
            grid[xMapPosition][yMapPosition] = -1
            canvas.delete("map")
            # OWN CODE END ===============================================================================================
            drawMap(canvas, grid)

    # update sensors on where charger is located in relation to the bot
    def senseCharger(self, registryPassives):
        lightL = 0.0
        lightR = 0.0

        for pp in registryPassives:
            if isinstance(pp, Charger):
                lx, ly = pp.getLocation()

                distanceL = math.sqrt((lx - self.sensorPositions[0]) * (lx - self.sensorPositions[0]) +
                                      (ly - self.sensorPositions[1]) * (ly - self.sensorPositions[1]))
                distanceR = math.sqrt((lx - self.sensorPositions[2]) * (lx - self.sensorPositions[2]) +
                                      (ly - self.sensorPositions[3]) * (ly - self.sensorPositions[3]))
                lightL = 200000 / (distanceL * distanceL)
                lightR = 200000 / (distanceR * distanceR)
        return lightL, lightR

    # update sensors on where bin is located in relation to the bot
    def senseBin(self, registryPassives):
        lightL = 0.0
        lightR = 0.0

        for pp in registryPassives:

            if isinstance(pp, Bin):
                lx, ly = pp.getLocation()

                distanceL = math.sqrt((lx - self.sensorPositions[0]) * (lx - self.sensorPositions[0]) +
                                      (ly - self.sensorPositions[1]) * (ly - self.sensorPositions[1]))
                distanceR = math.sqrt((lx - self.sensorPositions[2]) * (lx - self.sensorPositions[2]) +
                                      (ly - self.sensorPositions[3]) * (ly - self.sensorPositions[3]))
                lightL = 200000 / (distanceL * distanceL)
                lightR = 200000 / (distanceR * distanceR)
        return lightL, lightR

    def distanceTo(self, obj):
        xx, yy = obj.getLocation()
        return math.sqrt(math.pow(self.x - xx, 2) + math.pow(self.y - yy, 2))

    # collect nearby dirt and remove it from map
    def collectDirt(self, canvas, registryPassives, count):
        toDelete = []
        for idx, rr in enumerate(registryPassives):
            if isinstance(rr, Dirt):
                if self.distanceTo(rr) < 30 and self.currentDirt < self.capacity:
                    canvas.delete(rr.name)
                    toDelete.append(idx)
                    count.dirtCollected += 1
                    self.currentDirt += 1
        for ii in sorted(toDelete, reverse=True):
            del registryPassives[ii]
        return registryPassives

    # OWN CODE START ===============================================================================================

    def transferFunction(self, chargerL, chargerR, binL, binR, registryActives, algorithm, grid):
        # Object Avoidance - turn bot till it is able to avoid a collision with a wall or bot
        if self.collisionDetected:

            if not self.currentlyTurning:
                self.turning = random.randrange(10, 30)
                self.currentlyTurning = True

            if self.turning == 0 and self.currentlyTurning:
                self.currentlyTurning = False
                self.collisionDetected = False

            if self.currentlyTurning:
                self.vl = 2.0
                self.vr = -2.0
                self.turning -= 1

            if not self.currentlyTurning:
                self.collisionAvoided = True

        if not self.collisionDetected:

            if algorithm == "straight line":
                self.straightAlgorithm()
            elif algorithm == "random":
                self.randomAlgorithm()
            elif algorithm == "spiral":
                self.spiralAlgorithm()
            elif algorithm == "cell search":
                self.cellSearchAlgorithm(grid)
            elif algorithm == "cell search dirt":
                self.cellSearchDirtAlgorithm(grid)
            elif algorithm == "attract":
                self.attractAlgorithm(registryActives)
            elif algorithm == "repel":
                self.repelAlgorithm(registryActives)

            # when dirt capacity full move to bin
            if self.currentDirt == self.capacity:
                if binR > binL:
                    self.vl = 2.0
                    self.vr = -2.0
                elif binR < binL:
                    self.vl = -2.0
                    self.vr = 2.0
                if abs(binR - binL) < binL * 0.1:
                    self.vl = 5.0
                    self.vr = 5.0
                self.full = True

            if binL + binR > 200 and self.full:
                self.vl = 0.0
                self.vr = 0.0
                self.count = 0

            # when battery capacity low  move to charger
            if self.battery < 600:
                if chargerR > chargerL:
                    self.vl = 2.0
                    self.vr = -2.0
                elif chargerR < chargerL:
                    self.vl = -2.0
                    self.vr = 2.0
                if abs(chargerR - chargerL) < chargerL * 0.1:
                    self.vl = 5.0
                    self.vr = 5.0

                self.charge = True

            if chargerL + chargerR > 200 and self.charge:
                self.vl = 0.0
                self.vr = 0.0
                self.count = 0

    # Boids algorithm
    # Bots within a threshold move away from each other
    def repelAlgorithm(self, registryActives):
        nearestBot = None
        nearbyBots = []

        for bots in registryActives:

            if self.distanceTo(bots) < 400 and self.distanceTo(bots) != 0:
                nearbyBots.append(bots)
            nearestBot = None
            if nearbyBots:
                shortestDistance = 1000000.0
                for otherBot in nearbyBots:
                    d = self.distanceTo(otherBot)
                    if d < shortestDistance:
                        shortestDistance = d
                        nearestBot = otherBot

        self.vl = 5.0
        self.vr = 5.0

        if nearestBot is not None:
            self.noCollision = True
            fX = self.cameraPositions[4][0]
            fY = self.cameraPositions[4][1]

            if distanceToPoint(fX, fY, nearestBot.x, nearestBot.y) < self.distanceTo(nearestBot):
                if nearestBot.x - self.x == 0.0:
                    angle = math.atan((nearestBot.y - self.y) / 0.0001)
                else:
                    angle = math.atan((nearestBot.y - self.y) / (nearestBot.x - self.x))
                self.theta -= angle / 2.0

    # Boids algorithm
    # Nearby bots angle themselves towards the same direction and they move as a group
    def attractAlgorithm(self, registryActives):

        nearbyBots = []

        for bots in registryActives:

            if self.distanceTo(bots) < 150 and self.distanceTo(bots) != 0:
                nearbyBots.append(bots)

        self.vl = 5.0
        self.vr = 5.0

        averageAngle = 0.0
        if nearbyBots:
            for nb in nearbyBots:
                averageAngle += nb.theta
            averageAngle /= len(nearbyBots)
            self.theta = averageAngle

    # spiralling algorithm which moves in a circle outwards, if it collides with a wall or object,
    # it moves a set distance away till it can start spiralling again
    def spiralAlgorithm(self):

        if self.collisionAvoided:
            self.moving = 50
            self.count = 0
            self.collisionAvoided = False

        if self.moving > 0:
            self.vl = 5.0
            self.vr = 5.0
            self.moving -= 1

        elif self.moving <= 0 and self.currentlyTurning:
            self.vl = 5.0 + self.count
            self.vr = -5.0 + self.count
            self.count += 0.1
            self.currentlyTurning = False
        else:
            self.vl = 5.0
            self.vr = 5.0
            self.currentlyTurning = True

    # random wander algorithm which randomly turns and moves a random distance
    def randomAlgorithm(self):

        if self.currentlyTurning:
            self.vl = 2.0
            self.vr = -2.0
            self.turning -= 1
        else:
            self.vl = 5.0
            self.vr = 5.0
            self.moving -= 1
        if self.moving <= 0 and not self.currentlyTurning:
            self.turning = random.randrange(20, 30)
            self.currentlyTurning = True
        if self.turning == 0 and self.currentlyTurning:
            self.moving = random.randrange(50, 100)
            self.currentlyTurning = False

    # move in a straight line, used when creating the training set
    def straightAlgorithm(self):
        self.vl = 5.0
        self.vr = 5.0

    # Planning algorithm which uses grid to move towards nearby unvisited cells
    def cellSearchAlgorithm(self, grid):

        x = int(math.floor(self.x / 50))
        y = int(math.floor(self.y / 50))
        nearestSquare = None
        shortestDistance = 1000000.0

        for idx, xy in np.ndenumerate(grid):
            d = distanceToPoint(x, y, idx[0], idx[1])
            if d < shortestDistance and xy != -1:
                shortestDistance = d
                nearestSquare = idx

        self.calculateSensors(nearestSquare)

    # Planning algorithm which uses grid to move towards unvisited high dirt cells
    def cellSearchDirtAlgorithm(self, grid):

        x = int(math.floor(self.x / 50))
        y = int(math.floor(self.y / 50))
        nearestSquare = None
        shortestDistance = 1000000.0

        fourList = []
        threeList = []
        twoList = []
        oneList = []
        zeroList = []

        for idx, xy in np.ndenumerate(grid):

            if xy == 4:
                fourList.append(idx)
            elif xy == 3:
                threeList.append(idx)
            elif xy == 2:
                twoList.append(idx)
            elif xy == 1:
                oneList.append(idx)
            elif xy == 0:
                zeroList.append(idx)

        if fourList:

            for idx in fourList:
                d = distanceToPoint(x, y, idx[0], idx[1])
                if d < shortestDistance:
                    shortestDistance = d
                    nearestSquare = idx

        elif threeList:

            for idx in threeList:
                d = distanceToPoint(x, y, idx[0], idx[1])
                if d < shortestDistance:
                    shortestDistance = d
                    nearestSquare = idx

        elif twoList:

            for idx in twoList:
                d = distanceToPoint(x, y, idx[0], idx[1])
                if d < shortestDistance:
                    shortestDistance = d
                    nearestSquare = idx

        elif oneList:

            for idx in oneList:
                d = distanceToPoint(x, y, idx[0], idx[1])
                if d < shortestDistance:
                    shortestDistance = d
                    nearestSquare = idx

        else:

            for idx in zeroList:
                d = distanceToPoint(x, y, idx[0], idx[1])
                if d < shortestDistance:
                    shortestDistance = d
                    nearestSquare = idx

        self.calculateSensors(nearestSquare)

    # Move towards target location based on sensor sensitivity
    def calculateSensors(self, nearestSquare):

        target = (50 * nearestSquare[0] + 25, 50 * nearestSquare[1] + 25)

        if self.distanceToRightSensor(target[0], target[1]) > self.distanceToLeftSensor(target[0], target[1]):
            self.vl = 2.0
            self.vr = -2.0
        elif self.distanceToRightSensor(target[0], target[1]) < self.distanceToLeftSensor(target[0], target[1]):
            self.vl = -2.0
            self.vr = 2.0
        if abs(self.distanceToRightSensor(target[0], target[1]) - self.distanceToLeftSensor(target[0], target[1])) < self.distanceToLeftSensor(target[0], target[1]) * 0.5:
            self.vl = 5.0
            self.vr = 5.0
        if self.distanceToRightSensor(target[0], target[1]) > 75 or self.distanceToLeftSensor(target[0], target[1]) > 75 \
                and abs(self.distanceToRightSensor(target[0], target[1]) - self.distanceToLeftSensor(target[0], target[1])) < self.distanceToLeftSensor(target[0], target[1]) * 0.1:
            self.vl = 5.0
            self.vr = 5.0

    # OWN CODE END ===============================================================================================

    def distanceToRightSensor(self, lx, ly):
        return math.sqrt((lx - self.sensorPositions[0]) * (lx - self.sensorPositions[0]) +
                         (ly - self.sensorPositions[1]) * (ly - self.sensorPositions[1]))

    def distanceToLeftSensor(self, lx, ly):
        return math.sqrt((lx - self.sensorPositions[2]) * (lx - self.sensorPositions[2]) +
                         (ly - self.sensorPositions[3]) * (ly - self.sensorPositions[3]))

    def getLocation(self):
        return self.x, self.y

    # OWN CODE START ===============================================================================================

    # Collision detection used when creating the training set for danger threshold and logistic regression
    def collisionTraining(self, registryActives):
        collisionConfirmed = False

        for rr in registryActives:

            dd = self.distanceTo(rr)

            distance = (math.sqrt(1800) / 2) + 60

            if dd < distance and dd != 0:

                if distanceToPoint(self.cameraPositions[4][0], self.cameraPositions[4][1], rr.x, rr.y) < dd:
                    if not self.noCollision:
                        self.collisionDetected = True
                        collisionConfirmed = True

        return collisionConfirmed

    # Normal collision algorithm
    # compares the bot's location to nearby bots, if below a certain distance state a collision is detected
    def standardDetection(self, registryActives):

        for rr in registryActives:

            dd = self.distanceTo(rr)
            distance = math.sqrt(1800) + 60

            if dd < distance and dd != 0:

                if distanceToPoint(self.cameraPositions[4][0], self.cameraPositions[4][1], rr.x, rr.y) < dd:
                    self.collisionDetected = True

        collisionOccurred = self.checkIfCollision(registryActives)

        return collisionOccurred

    # Logistic Regression collision algorithm
    # if max sensor value is run through Logistic Regression classifier, and it outputs "danger" a collision is detected
    def logisticRegressionDetection(self, classifier, registryActives):

        sensors = self.look(registryActives)

        result = np.array([[max(sensors)]])
        result = ' '.join(classifier.predict(result))

        if result == "danger":
            self.collisionDetected = True

        collisionsOccurred = self.checkIfCollision(registryActives)

        return collisionsOccurred

    # Danger threshold collision algorithm
    # if max sensor value greater than the calculated threshold, a collision is detected
    def dangerThresholdDetection(self, dangerThresh, registryActives):

        sensors = self.look(registryActives)

        if max(sensors) > dangerThresh:
            self.collisionDetected = True

        collisionsOccurred = self.checkIfCollision(registryActives)

        return collisionsOccurred

    # Test collision detection of collision algorithms
    def checkIfCollision(self, registryActives):
        collisionsOccurred = False

        for rr in registryActives:

            if isinstance(rr, Bot):

                dd = self.distanceTo(rr)

                distance = (math.sqrt(1800) / 2) + 60

                if dd < distance and dd != 0:
                    if distanceToPoint(self.cameraPositions[4][0], self.cameraPositions[4][1], rr.x, rr.y) < dd:
                        collisionsOccurred = True

        return collisionsOccurred

    # OWN CODE END ===============================================================================================


# OWN CODE START ===============================================================================================
# Calculate distance between two points
def distanceToPoint(xx, yy, xx2, yy2):
    return math.sqrt(math.pow(xx2 - xx, 2) + math.pow(yy2 - yy, 2))


# OWN CODE END ===============================================================================================

# When a cell has been visited paint it yellow
def drawMap(canvas, grid):
    for xx in range(0, 20):
        for yy in range(0, 20):

            if grid[xx][yy] == -1:
                canvas.create_rectangle(50 * xx, 50 * yy, 50 * xx + 50, 50 * yy + 50, fill="yellow", width=0, tags="map")

    canvas.tag_lower("map")


# OWN CODE START ===============================================================================================
# Place 800 dirt in a random arrangement in the grid with cells having between 0 and 4 dirt.
# Dirt put in specific configurations within a cell
def placeDirt(registryPassives, canvas):
    # places dirt in a specific configuration
    dirtGrid = np.zeros((20, 20), dtype=np.int16)
    counter = 0
    for xx in range(20):
        for yy in range(20):

            if counter < 4:
                dirtGrid[xx][yy] = 4
            elif counter < 8:
                dirtGrid[xx][yy] = 3
            elif counter < 12:
                dirtGrid[xx][yy] = 2
            elif counter < 16:
                dirtGrid[xx][yy] = 1
            else:
                dirtGrid[xx][yy] = 0
        counter += 1

    dirtGrid = list(chain.from_iterable(dirtGrid))
    random.shuffle(dirtGrid)
    dirtGrid = np.array(dirtGrid).reshape(-1, 20)

    i = 0
    for xx in range(20):
        for yy in range(20):
            for val in range(dirtGrid[xx][yy]):

                if val == 0:
                    dirtX = 50 * xx + 35
                    dirtY = 50 * yy + 15
                elif val == 1:
                    dirtX = 50 * xx + 15
                    dirtY = 50 * yy + 15
                elif val == 2:
                    dirtX = 50 * xx + 15
                    dirtY = 50 * yy + 35
                else:
                    dirtX = 50 * xx + 35
                    dirtY = 50 * yy + 35

                if dirtGrid[xx][yy] != 0:
                    dirt = Dirt("Dirt" + str(i), dirtX, dirtY)
                    registryPassives.append(dirt)
                    dirt.draw(canvas)
                i += 1

    return dirtGrid


# OWN CODE END ===============================================================================================


def buttonClicked(x, y, bot):
    bot.x = x
    bot.y = y
    bot.theta += 0.05


def initialise(window):
    window.resizable(False, False)
    canvas = tk.Canvas(window, width=1000, height=1000)
    canvas.pack()
    return canvas


def register(canvas, noOfBots):
    registryActives = []
    registryPassives = []

    # OWN CODE START ===============================================================================================
    # Randomly generated bot starting positions
    positions = [(75 + random.randrange(1, 100), 75 + random.randrange(1, 100), random.uniform(0, math.pi * 2)),
                 (925 - random.randrange(1, 100), 75 + random.randrange(1, 100), random.uniform(0, math.pi * 2)),
                 (75 + random.randrange(1, 100), 925 - random.randrange(1, 100), random.uniform(0, math.pi * 2)),
                 (925 - random.randrange(1, 100), 925 - random.randrange(1, 100), random.uniform(0, math.pi * 2)),
                 (75 + random.randrange(1, 100), 500 + random.randrange(1, 100), random.uniform(0, math.pi * 2)),
                 (925 - random.randrange(1, 100), 500 - random.randrange(1, 100), random.uniform(0, math.pi * 2)),
                 (500 + random.randrange(1, 100), 925 - random.randrange(1, 100), random.uniform(0, math.pi * 2)),
                 (500 - random.randrange(1, 100), 75 + random.randrange(1, 100), random.uniform(0, math.pi * 2)),
                 (450 - random.randrange(1, 100), 450 + random.randrange(1, 100), random.uniform(0, math.pi * 2)),
                 (550 + random.randrange(1, 100), 550 - random.randrange(1, 100), random.uniform(0, math.pi * 2))]

    random.shuffle(positions)

    # OWN CODE END ===============================================================================================

    for i in range(0, noOfBots):
        bot = Bot("Bot" + str(i), positions[i])
        registryActives.append(bot)
        bot.draw(canvas)
    charger = Charger("Charger")
    registryPassives.append(charger)
    charger.draw(canvas)
    bin1 = Bin("Bin")
    registryPassives.append(bin1)
    bin1.draw(canvas)
    # Create dirt map to be used by planning algorithms
    dirtMap = placeDirt(registryPassives, canvas)
    count = Counter()
    canvas.bind("<Button-1>", lambda event: buttonClicked(event.x, event.y, registryActives[0]))
    return registryActives, registryPassives, count, dirtMap


def moveIt(canvas, registryActives, registryPassives, count, window, algorithm, grid, classifier, dangerThresh):
    count.totalMoves += 1

    for rr in registryActives:
        chargerIntensityL, chargerIntensityR = rr.senseCharger(registryPassives)
        binIntensityL, binIntensityR = rr.senseBin(registryPassives)
        rr.transferFunction(chargerIntensityL, chargerIntensityR, binIntensityL, binIntensityR, registryActives, algorithm, grid)
        rr.move(canvas, registryPassives, 1.0, grid, count)
        registryPassives = rr.collectDirt(canvas, registryPassives, count)
        # OWN CODE START ===============================================================================================
        # use collision algorithm specified by the user
        if classifier is None and dangerThresh is None:
            collision = rr.standardDetection(registryActives)
        elif dangerThresh is None:
            collision = rr.logisticRegressionDetection(classifier, registryActives)
        else:
            collision = rr.dangerThresholdDetection(dangerThresh, registryActives)
        # Record the number of collisions occurring
        if collision:
            count.totalCollisions += 1

        visitedCells = np.count_nonzero(count.visitedGrid)
        # OWN CODE END ===============================================================================================
        # End run if moves over threshold or all cells have been visited or all dirt has been collected
        if count.totalMoves > 1000 or visitedCells == 400 or count.dirtCollected == 800:
            window.destroy()
            return

    canvas.after(20, moveIt, canvas, registryActives, registryPassives, count, window, algorithm, grid, classifier, dangerThresh)

# Collect collision data for training set till 20000 danger values have been stored
def trainIt(canvas, registryActives, registryPassives, count, window, trainingSet):
    for rr in registryActives:
        seen = rr.look(registryActives)
        chargerIntensityL, chargerIntensityR = rr.senseCharger(registryPassives)
        binIntensityL, binIntensityR = rr.senseBin(registryPassives)
        rr.transferFunction(chargerIntensityL, chargerIntensityR, binIntensityL, binIntensityR, registryActives, "straight line", None)
        rr.move(canvas, registryPassives, 1.0, None, None)
        collision = rr.collisionTraining(registryActives)
        if collision and max(seen) != 0:
            count.totalCollisions += 1
        trainingSet.append((seen, collision))

        if count.totalCollisions >= 20000:
            window.destroy()
            return
    canvas.after(0, trainIt, canvas, registryActives, registryPassives, count, window, trainingSet)


# OWN CODE START ===============================================================================================
# Create training dataset and export it to text files for re-usability
def train(noOfBots):
    window = tk.Tk()
    canvas = initialise(window)
    registryActives, registryPassives, count, dirtMap = register(canvas, noOfBots)
    trainingSet = []
    trainIt(canvas, registryActives, registryPassives, count, window, trainingSet)
    window.mainloop()

    dangerFile = open("dangerValues.txt", 'w')
    safeFile = open("safeValues.txt", 'w')
    counterDanger = 0
    counterSafe = 0

    for idx, tt in enumerate(trainingSet):

        if counterDanger >= 20000:
            break

        if tt[1] is False and counterSafe < 20000:
            safeFile.write(str(max(trainingSet[idx][0])))
            safeFile.write("\n")
            counterSafe += 1

        elif tt[1] is True and max(tt[0]) != 0:
            dangerFile.write(str(max(trainingSet[idx][0])))
            dangerFile.write("\n")
            counterDanger += 1


# Read in dataset to train Logistic Regression model
def logisticRegression():
    dangerValues = []
    safeValues = []
    intentLabels = []
    dangerFile = open("dangerValues.txt", 'r')
    safeFile = open("safeValues.txt", 'r')
    dangerLines = dangerFile.readlines()
    safeLines = safeFile.readlines()

    for line in safeLines:
        if line != "\n":
            safeValues.append([float(line)])

    for line in dangerLines:
        if line != "\n":
            dangerValues.append([float(line)])

    for _ in safeValues:
        intentLabels.append("safe")

    for _ in dangerValues:
        intentLabels.append("danger")

    x_train, x_test, y_train, y_test = train_test_split(safeValues + dangerValues, intentLabels, stratify=intentLabels, random_state=11)
    classifier = LogisticRegression(random_state=0).fit(x_train, y_train)
    model_accuracy(classifier, x_test, y_test)

    return classifier


# Tests Logistic Regression model with our test data to see how well our classifier predicts a collision
def model_accuracy(classifier, x_test, y_test):

    intent_predict = classifier.predict(x_test)
    print("Accuracy score: ", accuracy_score(y_test, intent_predict))
    print("Safe F1 Score: ", f1_score(y_test, intent_predict, average="binary", pos_label="safe"))
    print("Danger F1 Score: ", f1_score(y_test, intent_predict, average="binary", pos_label="danger"))
    print("Safe Precision score: ", precision_score(y_test, intent_predict, average="binary", pos_label="safe"))
    print("Danger Precision score: ", precision_score(y_test, intent_predict, average="binary", pos_label="danger"))
    print("Safe Recall score: ", recall_score(y_test, intent_predict, average="binary", pos_label="safe"))
    print("Danger Recall score: ", recall_score(y_test, intent_predict, average="binary", pos_label="danger"))
    print("Confusion Matrix: \n", confusion_matrix(y_test, intent_predict))


# Read in dataset to calculate danger threshold value
def dangerThreshold():
    danger = 0.0
    count = 0
    dangerFile = open("dangerValues.txt", 'r')
    dangerLines = dangerFile.readlines()

    for line in dangerLines:
        if line != "\n":
            danger += float(line)
            count += 1

    danger /= count

    return danger


def main(noOfBots, algorithm, collisionAlg):
    if collisionAlg is not None:
        if collisionAlg == "logisticRegression":
            if os.path.isfile("classifier.joblib"):
                classifier = load("classifier.joblib")
            elif not os.path.isfile("dangerValues.txt") or not os.path.isfile("safeValues.txt"):
                train(noOfBots)
                classifier = logisticRegression()
                dump(classifier, "classifier.joblib")
            else:
                classifier = logisticRegression()
                dump(classifier, "classifier.joblib")

            dangerThresh = None
        else:
            if not os.path.isfile("dangerValues.txt"):
                train(noOfBots)

            dangerThresh = dangerThreshold()
            classifier = None
    else:
        classifier = None
        dangerThresh = None

    window = tk.Tk()
    canvas = initialise(window)
    registryActives, registryPassives, count, dirtMap = register(canvas, noOfBots)
    count.totalCollisions = 0
    moveIt(canvas, registryActives, registryPassives, count, window, algorithm, dirtMap, classifier, dangerThresh)
    window.mainloop()

    return count.dirtCollected, count.visitedGrid, count.totalCollisions


# main(2, "random", "dangerThreshold")

# OWN CODE END ===============================================================================================
