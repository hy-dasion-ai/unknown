import numpy as np

""" 
getAllCriticals
Given data and a predefined threshold, find all critical points (e.g. min/max)

(later) 
Utilize libraries: https://github.com/MonsieurV/py-findpeaks 
Optimize using np.argsort(), serialize 
""" 
def getAllCriticals(axis, distThreshold):
	maxIndex = axis.idxmax()
	minIndex = axis.idxmin()
	
	#boolean to record orientation of global max and min
	leftMin = minIndex < maxIndex

	absStart = 0
	absEnd = len(axis) - 1

	#if global min then global max, iterate over left of min and right of max
	#otherwise, do opposite
	if leftMin:
		leftCrits = findCritsLeft(axis, minIndex, absStart, False, distThreshold)
		rightCrits = findCritsRight(axis, maxIndex, absEnd, True, distThreshold)
	else:
		leftCrits = findCritsLeft(axis, maxIndex, absStart, True, distThreshold)
		rightCrits = findCritsRight(axis, minIndex, absEnd, False, distThreshold)

	#append into numpy array
	c = np.append(leftCrits, rightCrits)
	
	#return all critical points
	return c

"""
findCritsLeft
Given data, extract the critical values (e.g. mins and maxes) to the left of given point

axis = data
end = starting point
absStart = absolute start of data
isMax = boolean for whether the starting point is max or min
"""
def findCritsLeft(axis, end, absStart, isMax, distThreshold):
	#returns all the critical values to the left of given "end"
	leftCrits = [end]
	while (end != absStart): 
		temp = axis[end]
		if (isMax):
			nextMin = axis[:end].idxmin()
			leftCrits.append(nextMin)
			isMax = False
			end = nextMin
		else:
			nextMax = axis[:end].idxmax()
			leftCrits.append(nextMax)
			isMax = True
			end = nextMax
		if (abs(float(axis[end]) - float(temp)) < distThreshold):
			break
	
	leftCrits.append(absStart)

	leftCrits.reverse()
	return leftCrits

"""
findCritsRight
Given data, extract the critical values (e.g. mins and maxes) to the right of given point

axis = data
start = starting point
absEnd = absolute end of data
isMax = boolean for whether the starting point is max or min
"""
def findCritsRight(axis, start, absEnd, isMax, distThreshold):
	#returns all the critical values to the right of the given "start"
	rightCrits = [start]
	while (start != absEnd):

		temp = axis[start]
		if (isMax):
			nextMin = axis[start:].idxmin()
			rightCrits.append(nextMin)
			isMax = False
			start = nextMin
		else:
			nextMax = axis[start:].idxmax()
			rightCrits.append(nextMax)
			isMax = True
			start = nextMax
		if (abs(float(axis[start]) - float(temp)) < distThreshold):
			break


	rightCrits.append(absEnd)

	return rightCrits


"""
calcDistanceWrapper
Wrapper for function 'calcDistance'

This allows easy way to assess series of data to a single projected 2D line
"""
def calcDistanceWrapper(xs, ys, x1, y1, x2, y2):
	distances = []

	for x, y in zip(xs, ys):
		distances.append(calcDistance(float(x), float(y), float(x1), float(y1), float(x2), float(y2)))

	return distances

"""
calcDistance
Given a point, and two additional points that define a line, find min dist
Utilizes closed form to calculate distance

function signature: calcDistance(point, line point 1, line point2)
"""
def calcDistance(x, y, x1, y1, x2, y2):
	numerator = abs((y2-y1)*(x) - (x2-x1)*y + x2*y1 - y2*x1)
	denominator = ((y2-y1)**2+(x2-x1)**2)**(1/2)
	return numerator/denominator

"""
getVectorsNum
Given an axis (e.g. x-axis from gyroscope) and the timestamps, run GUPR_D

Hyperparameters:
curveThreshold: the maxDist allowed for projection of real data onto line segment
distThreshold: the similarity measure upon which we assume local max approximately equals local min
numVectors: specify the number of the resulting vector

Return:
1. rebased = matrix containing the vectors (set to origin)
2. vectorPoints = indices (in original data) of the points from which vectors are constructed
3. criticals = indices of all critical points (global/local max/min) in time-order
"""
def getVectorsNum(axis, time, curveThreshold = 0.5, distThreshold = 0.5, numVectors = False):
	#get all the criticals: should contain abs start and abs end of time series
	criticals = getAllCriticals(axis, distThreshold)

	#all the indices for where the vector points are
	vectorPoints = []

	stack = list(np.flip(criticals, 0))
	weights = dict()

	while stack:
		index1 = stack.pop()
		#end condition:
		if (index1 == criticals[-1]):
			vectorPoints.append(index1)
			break
		index2 = stack.pop()

		#skip condition:
		if (abs(index1-index2) < 2):
			vectorPoints.append(index1)
			stack.append(index2)
			continue

		# 1. evaluate distance of "real data" to "linear line"
		distances = calcDistanceWrapper(time[index1:index2], axis[index1:index2], \
			time[index1], axis[index1], time[index2], axis[index2])
		# 2. get max dist point
		maxDistIndex = np.argmax(distances)

		# 3. check threshold
		# 3.b. get weight and store if it doesn't pass "the first time"
		weight = distances[maxDistIndex]
		if (weight < curveThreshold):
			vectorPoints.append(index1)
			stack.append(index2)
			continue
		else:
			weights[(index1, index2)] = weight
			stack.append(index2)
			stack.append(index1 + maxDistIndex)
			stack.append(index1)

	#if a number of vectors were specified, then refer to the weights and recreate
	if (numVectors):
		#sort such that the first interval (start, end) has the lowest weight 
		sortedIntervals = sorted(weights, key = weights.get)
		currentCount = len(vectorPoints) - 1

		if (currentCount < numVectors):
			return "ERROR: current number of vectors already smaller than specified number"

		#FUCK: consider cases in which the next interval removes too many!
		setOfVectorPoints = set(vectorPoints)
		for start, end in sortedIntervals:
			#currentCount = len(setOfVectorPoints)
			
			if (currentCount == numVectors):
				break

			if (currentCount < numVectors):
				print("ERROR: Too many vectors removed")
				return

			if start in setOfVectorPoints and end in setOfVectorPoints:
				#setOfVectorPoints = set([value for value in setOfVectorPoints if (value <= start or value >= end)])
				toRemove = set()
				for value in setOfVectorPoints:
					if (value > start and value < end):
						toRemove.add(value)
						currentCount -= 1
				setOfVectorPoints -= toRemove


		vectorPoints = sorted(list(setOfVectorPoints))

	#######################################################
	#return vectorPoints
	#######################################################

	#visualize to check
	# plt.plot(time, axis, label = "Original Data")
	# plt.plot(time[vectorPoints], axis[vectorPoints], label = "Vectors")
	# plt.title("Comparison of Original to Linear Segment Approximation")
	# plt.show()
	
	#construct the vectors from indicies
	matrix = np.zeros(shape = (len(vectorPoints), 2))
	for i, index in enumerate(vectorPoints):
		matrix[i][0] = time[index]
		matrix[i][1] = axis[index]

	#rebase the vectors to origin
	rebased = []
#	newMatrix = [matrix[i]]
	newMatrix = []
	#first comparison is with the very first point
	base = np.asarray([time[0], axis[0]])
	for i, vector in enumerate(matrix[1:]):
		diff = vector - base
		if (not all(diff == [0, 0])):
			rebased.append(diff)
			newMatrix.append(vector)
		base = vector

	matrix = np.asarray(newMatrix)
	rebased = np.asarray(rebased)
	# if (len(rebased)+1 != len(matrix)):
	# 	print(len(rebased), len(matrix))

#	return rebased, matrix, criticals
	return rebased, matrix, vectorPoints
