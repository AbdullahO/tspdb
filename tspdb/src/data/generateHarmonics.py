######################################################
#
# Generate Harmonics data
#
######################################################
import numpy as np

def generate(sineCoeffArray, sinePeriodsArray, cosineCoeffArray, cosinePeriodsArray, timeSteps, tStart = 0):

	if (len(sineCoeffArray) != len(sinePeriodsArray)):
		raise Exception('sineCoeffArray and sinePeriodsArray must be of the same length.')

	if (len(cosineCoeffArray) != len(cosinePeriodsArray)):
		raise Exception('cosineCoeffArray and cosinePeriodsArray must be of the same length.')

	outputArray = np.zeros(timeSteps)
	T = float(timeSteps)
	for i in range(tStart, timeSteps):
		value = 0.0
		for j in range(0, len(sineCoeffArray)):
			value += (sineCoeffArray[j] * np.sin(i * sinePeriodsArray[j] * 2.0 * np.pi / T ))

		for k in range(0, len(cosineCoeffArray)):
			value += (cosineCoeffArray[k] * np.cos(i * cosinePeriodsArray[k] * 2.0 * np.pi / T))

		outputArray[i] = value

	return outputArray