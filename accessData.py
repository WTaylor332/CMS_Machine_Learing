import numpy as np
import uproot
from tqdm import tqdm

def loadData(dataFile):
    f = uproot.open(dataFile+':L1TrackNtuple/eventTree')
    events = f.arrays(['trk_pt', 'trk_eta','trk_phi','trk_z0','trk_nstub', 'trk_bendchi2','trk_chi2dof','pv_MC','pv_L1reco'])
    return events['trk_z0'], events['trk_pt'], events['pv_MC']


def rawPaddedData(eventZ, eventPT):
    padSize = 300
    eventsSorted = np.array([[]])

    for i in range(len(eventZ)):
        indexSort = np.argsort(eventZ[i])
        zSort = np.sort(eventZ[i])
        padding = np.array([999999 for x in range((padSize-len(zSort)))])
        zSort = np.append(zSort, padding)
        ptSort = np.array([eventPT[i][indexSort]])
        ptSort = np.append(ptSort, padding)
        eventsSorted = np.append(eventsSorted, [zSort, ptSort])

    eventsSorted = eventsSorted.reshape(len(eventZ),2,padSize)

    return eventsSorted

# binning

def histogramData(rawData):

    binnedMatrix = np.zeros((rawData.shape[0], rawData.shape[1], rawData.shape[2]))
    ptBin = np.zeros(300) + 999999

    for i in range(rawData.shape[0]):
        ptBin = np.zeros(300) + 999999
        binnedMatrix[i][0] = np.linspace(-14.9,15,300).round(2)
        ptValue = 0
        binValue = -14.9
        count = 0 
        while binValue <= 15:
            if rawData[i][0][count] <= binValue:
                ptValue += (rawData[i][1][count])
                count += 1
            if rawData[i][0][count] > binValue:
                index = np.where(binnedMatrix[i][0] == round(binValue,1))
                ptBin[index] = ptValue
                binValue += 0.1
                ptValue = 0

        binnedMatrix[i][1] = ptBin
    return binnedMatrix
