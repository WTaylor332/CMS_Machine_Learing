import numpy as np
import uproot
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def loadData(dataFile):
    f = uproot.open(dataFile+':L1TrackNtuple/eventTree')
    events = f.arrays(['trk_pt', 'trk_eta','trk_phi','trk_z0','trk_nstub', 'trk_bendchi2','trk_chi2dof','pv_MC','pv_L1reco'])
    return events['trk_z0'], events['trk_pt'], events['pv_MC']
  

def rawPaddedData(eventZ, eventPT):
    padSize = 300
    eventZScaled = np.array([[]])
    eventPTScaled = np.array([[]])
    for i in range(len(eventZ)):
        indexSort = np.argsort(eventZ[i])
        zSort = np.sort(eventZ[i])
        padding = np.array([np.nan for x in range((padSize-len(zSort)))])
        zSort = np.append(zSort, padding)
        ptSort = np.array([eventPT[i][indexSort]])
        ptSort = np.append(ptSort, padding)
        eventZScaled = np.append(eventZScaled, [zSort])
        eventPTScaled = np.append(eventPTScaled, [ptSort])

    arr = np.stack((eventZScaled, eventPTScaled), axis=1)

    scaler = StandardScaler().fit(arr)
    arrScaled = scaler.transform(arr)
    eventPTScaled = arrScaled[:,1].reshape(len(eventPT), padSize)

    return eventZScaled.reshape(1360,300), eventPTScaled



# binning

def histogramData(z, pt):
    binnedMatrix = np.zeros((z.shape[0], z.shape[1]))

    for i in range(z.shape[0]):
        ptBin = np.zeros(300)
        binValueMatrix = np.linspace(-14.9,15,300).round(2)
        ptValue = 0
        binValue = -14.9
        count = 0 
        while binValue <= 15:
            if z[i][count] <= binValue:
                ptValue += (pt[i][count])
                count += 1
            else:
                index = np.where(binValueMatrix == round(binValue,1))
                ptBin[index] = ptValue
                binValue += 0.1
                ptValue = 0

        binnedMatrix[i] = ptBin
    return binnedMatrix


name = "TTbar.root"
n = "GTTObjects_ttbar200PU_1.root"
eventZ,  eventPT, eventPV = loadData(name)
print("s")
print(len(eventZ), len(eventPT), len(eventPV))
print(len(eventZ[0]), len(eventPT[0]), len(eventPV[0]))
zRaw, ptRaw = rawPaddedData(eventZ, eventPT)
print("w")
ptBin = histogramData(zRaw, ptRaw)

np.savez('TTbar', z=zRaw, pt=ptRaw, ptB=ptBin, pv=np.array(eventPV))
