from time import clock
import sys
sys.path.append('../../../')
print (sys.path)
from tspdb.src.data import generateHarmonics as gH
from  tspdb.src.data import generateTrend as gT
import tspdb.src.data.generateARMA as gA
import numpy as np
from tspdb.src.hdf_util import write_data
import matplotlib.pyplot as plt

def armaDataTest(timeSteps):
    arLags = []#[0.4, 0.3, 0.2]
    maLags = []#[0.5, 0.1]
    startingArray = np.zeros(np.max([len(arLags), len(maLags)])) # start with all 0's
    noiseMean = 0.0
    noiseSD = [1.0]
    (observedArray, meanArray, errorArray) = gA.generate(arLags, maLags, startingArray, timeSteps, noiseMean, noiseSD)
    return (observedArray, meanArray)

def trendDataTest(timeSteps):
    dampening = 2.0*float(1.0/timeSteps)
    power = 0.35
    displacement = -2.5
    f1 = gT.linearTrendFn
    data = gT.generate(f1, power=power, displacement=displacement, timeSteps=timeSteps)
    f2 = gT.logTrendFn
    f3 = gT.negExpTrendFn
    return data


def harmonicDataTest(timeSteps):
    sineCoeffs = [-2.0, 3.0]
    sinePeriods = [560.0, 30.0]
    cosineCoeffs = [-2.5]
    cosinePeriods = [16.0]
    data = gH.generate(sineCoeffs, sinePeriods, cosineCoeffs, cosinePeriods, timeSteps)
    #plt.plot(data)
    #plt.show()
    return data


timeSteps = 10**5 +10000



print('generating data..')
dt = clock()
harmonicsTS = harmonicDataTest(timeSteps)
trendTS = trendDataTest(timeSteps)
(armaTS, armaMeanTS) = armaDataTest(timeSteps)

meanTS = harmonicsTS + trendTS #+ armaMeanTS
# combinedTS = harmonicsTS + trendTS + armaTS
var = harmonicsTS
var = (var - min(var))
errorArray = np.random.normal(0, np.sqrt(var[:timeSteps]), timeSteps)
combinedTS = meanTS + errorArray
# max1 = np.nanmax(combinedTS)
# min1 = np.nanmin(combinedTS)
# max2 = np.nanmax(meanTS)
# min2 = np.nanmin(meanTS)
# max = np.max([max1, max2])
# min = np.min([min1, min2])

# combinedTS = tsUtils.normalize(combinedTS, max, min)
# meanTS = tsUtils.normalize(meanTS, max, min)
# p = 1
plt.plot(combinedTS, label = 'obs')
plt.plot(meanTS, label = 'mean')
plt.plot(var, label = 'var')
plt.show()
print('Data Generated in ', clock() - dt)

write_data('MixtureTS_var2.h5', 'means', meanTS)
write_data('MixtureTS_var2.h5', 'obs', combinedTS,'a')
write_data('MixtureTS_var2.h5', 'var', var,'a')
# DF = pd.DataFrame()
# DF['means'] = meanTS
# DF['Obs'] =  combinedTS
# DF['trainData'] =  trainData
# DF.to_hdf('MixtureTS.h5','ts1')
