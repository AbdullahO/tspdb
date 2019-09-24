#############################################################
#
# Robust Synthetic Control Tests (based on ALS)
#
# You need to ensure that this script is called from
# the tslib/ parent directory or tslib/tests/ directory:
#
# 1. python tests/testScriptSynthControlALS.py
# 2. python testScriptSynthControlALS.py
#
#############################################################
import sys, os
sys.path.append("../..")
sys.path.append("..")
sys.path.append(os.getcwd())

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import copy

from tslib.src import tsUtils
from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from tslib.tests import testdata

def basque(filename):

	# BASQUE COUNTRY STUDY
	df = pd.read_csv(filename)
	pivot = df.pivot_table(values='gdpcap', index='regionname', columns='year')
	pivot = pivot.drop('Spain (Espana)')
	dfBasque = pd.DataFrame(pivot.to_records())

	allColumns = dfBasque.columns.values

	states = list(np.unique(dfBasque['regionname']))
	years = np.delete(allColumns, [0])

	basqueKey = 'Basque Country (Pais Vasco)'
	states.remove(basqueKey)
	otherStates = states

	yearStart = 1955
	yearTrainEnd = 1971
	yearTestEnd = 1998

	singvals = 1
	p = 1.0

	trainingYears = []
	for i in range(yearStart, yearTrainEnd, 1):
		trainingYears.append(str(i))

	testYears = []
	for i in range(yearTrainEnd, yearTestEnd, 1):
		testYears.append(str(i))

	trainDataMasterDict = {}
	trainDataDict = {}
	testDataDict = {}
	for key in otherStates:
		series = dfBasque[dfBasque['regionname'] == key]
		
		trainDataMasterDict.update({key: series[trainingYears].values[0]})

		# randomly hide training data
		(trainData, pObservation) = tsUtils.randomlyHideValues(copy.deepcopy(trainDataMasterDict[key]), p)
		trainDataDict.update({key: trainData})
		testDataDict.update({key: series[testYears].values[0]})

	series = dfBasque[dfBasque['regionname'] == basqueKey]
	trainDataMasterDict.update({basqueKey: series[trainingYears].values[0]})
	trainDataDict.update({basqueKey: series[trainingYears].values[0]})
	testDataDict.update({basqueKey: series[testYears].values[0]})

	trainMasterDF = pd.DataFrame(data=trainDataMasterDict)
	trainDF = pd.DataFrame(data=trainDataDict)
	testDF = pd.DataFrame(data=testDataDict)

	# model
	rscModel = RobustSyntheticControl(basqueKey, singvals, len(trainDF), probObservation=1.0,  modelType='als', otherSeriesKeysArray=otherStates)

	# fit the model
	rscModel.fit(trainDF)

	# save the denoised training data
	denoisedDF = rscModel.model.denoisedDF()

	# predict - all at once
	predictions = rscModel.predict(testDF)
	
	# plot
	yearsToPlot = range(yearStart, yearTestEnd, 1)
	interventionYear = yearTrainEnd - 1
	plt.plot(yearsToPlot, np.append(trainMasterDF[basqueKey], testDF[basqueKey], axis=0), color='red', label='observations')
	plt.plot(yearsToPlot, np.append(denoisedDF[basqueKey], predictions, axis=0), color='blue', label='predictions')
	plt.axvline(x=interventionYear, linewidth=1, color='black', label='Intervention')
	plt.ylim((0, 12))
	legend = plt.legend(loc='lower right', shadow=True)
	plt.title('Abadie et al. Basque Country Case Study - $p = %.2f$' %p)
	plt.show()

def prop99(filename):

	# CALIFORNIA PROP 99 STUDY
	df = pd.read_csv(filename)
	df = df[df['SubMeasureDesc'] == 'Cigarette Consumption (Pack Sales Per Capita)']
	pivot = df.pivot_table(values='Data_Value', index='LocationDesc', columns=['Year'])
	dfProp99 = pd.DataFrame(pivot.to_records())

	allColumns = dfProp99.columns.values

	states = list(np.unique(dfProp99['LocationDesc']))
	years = np.delete(allColumns, [0])

	caStateKey = 'California'
	states.remove(caStateKey)
	otherStates = states

	yearStart = 1970
	yearTrainEnd = 1989
	yearTestEnd = 2015

	singvals = 2
	p = 1.0

	trainingYears = []
	for i in range(yearStart, yearTrainEnd, 1):
		trainingYears.append(str(i))

	testYears = []
	for i in range(yearTrainEnd, yearTestEnd, 1):
		testYears.append(str(i))

	trainDataMasterDict = {}
	trainDataDict = {}
	testDataDict = {}
	for key in otherStates:
		series = dfProp99[dfProp99['LocationDesc'] == key]
		
		trainDataMasterDict.update({key: series[trainingYears].values[0]})

		# randomly hide training data
		(trainData, pObservation) = tsUtils.randomlyHideValues(copy.deepcopy(trainDataMasterDict[key]), p)
		trainDataDict.update({key: trainData})
		testDataDict.update({key: series[testYears].values[0]})

	series = dfProp99[dfProp99['LocationDesc'] == caStateKey]
	trainDataMasterDict.update({caStateKey: series[trainingYears].values[0]})
	trainDataDict.update({caStateKey: series[trainingYears].values[0]})
	testDataDict.update({caStateKey: series[testYears].values[0]})

	trainMasterDF = pd.DataFrame(data=trainDataMasterDict)
	trainDF = pd.DataFrame(data=trainDataDict)
	testDF = pd.DataFrame(data=testDataDict)

	# model
	rscModel = RobustSyntheticControl(caStateKey, singvals, len(trainDF), probObservation=1.0, modelType='als', otherSeriesKeysArray=otherStates)

	# fit the model
	rscModel.fit(trainDF)

	# save the denoised training data
	denoisedDF = rscModel.model.denoisedDF()

	# predict - all at once
	predictions = rscModel.predict(testDF)
	
	# plot
	yearsToPlot = range(yearStart, yearTestEnd, 1)
	interventionYear = yearTrainEnd - 1
	plt.plot(yearsToPlot, np.append(trainMasterDF[caStateKey], testDF[caStateKey], axis=0), color='red', label='observations')
	plt.plot(yearsToPlot, np.append(denoisedDF[caStateKey], predictions, axis=0), color='blue', label='predictions')
	plt.axvline(x=interventionYear, linewidth=1, color='black', label='Intervention')
	legend = plt.legend(loc='lower left', shadow=True)
	plt.title('Abadie et al. Prop 99 Case Study (CA) - $p = %.2f$' %p)
	plt.show()


def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    print("-------------------------------------------------------")
    print("---------- Robust Synthetic Control (ALS). ------------------")
    print("-------------------------------------------------------")

    directory = os.path.dirname(testdata.__file__)

    print("    Proposition 99 (California)     ")
    
    prop99Filename = directory + '/prop99.csv'
    prop99(prop99Filename)

    print("    Basque Country                  ")
    basqueFilename = directory + '/basque.csv'
    basque(basqueFilename)
    
    print("-------------------------------------------------------")
    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":
    main()