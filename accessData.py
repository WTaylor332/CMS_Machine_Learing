import numpy as np
import uproot
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def loadData(dataFile):
    f = uproot.open(dataFile+':L1TrackNtuple/eventTree')
    events = f.arrays(['trk_pt', 'trk_eta','trk_phi','trk_z0','trk_nstub', 'trk_bendchi2','trk_chi2dof','pv_MC','pv_L1reco'])
    return events['trk_z0'], events['trk_pt'], events['pv_MC']
  

def modifyData(z, pt, ptCut=2):

    pt = np.where(pt<2, np.nan, pt)
    index = np.argwhere(np.isnan(pt))

    infIndex = np.argwhere(np.isinf(pt))
    pt[infIndex] = np.nan

    index = np.array(index).flatten()
    z[index] = np.nan
    z[infIndex] = np.nan

    return z, pt


def rawPaddedData(eventZ, eventPT):
    padSize = 400
    eventZData = np.zeros((len(eventZ), padSize))
    eventPTData = np.zeros((len(eventPT), padSize))
    trackLength = np.zeros(len(eventZ))
    maxTLength = 0
    for i in tqdm(range(len(eventZ))):

        zSort = np.zeros(padSize)
        padding = np.array([np.nan for x in range((padSize-len(eventZ[i])))])
        zSort[:len(eventZ[i])] = eventZ[i]
        zSort[len(eventZ[i]):] = padding

        ptSort = np.zeros(padSize)
        ptSort[:len(eventPT[i])] = eventPT[i]
        ptSort[len(eventPT[i]):] = padding

        z, pt = modifyData(zSort, ptSort)

        indexSort = np.argsort(z)
        zSort = np.sort(z)
        ptSort = pt[indexSort]

        if np.argwhere(np.isnan(zSort))[0][0] > maxTLength:
            maxTLength = np.argwhere(np.isnan(zSort))[0][0]
            trackLength[i] = np.argwhere(np.isnan(zSort))[0][0]

        eventZData[i] = zSort
        eventPTData[i] = ptSort

    # to scale pt data
    zFlat = eventZData.flatten()
    ptFlat = eventPTData.flatten()
    arr = np.stack((zFlat, ptFlat), axis=1)
    print(arr.shape)
    scaler = StandardScaler().fit(arr)
    arrScaled = scaler.transform(arr)
    eventPTData = arrScaled[:,1].reshape(len(eventPT), padSize)
    eventZData = eventZData[:, :maxTLength]
    eventPTData = eventPTData[:, :maxTLength]
    print(eventZData.shape, eventPTData.shape)
    return eventZData, eventPTData, trackLength, maxTLength


# binning

def histogramData(z, pt):
    ptBinnedMatrix = np.zeros((z.shape[0], 300))
    trackBinnedMatrix = np.zeros((z.shape[0], 300))
    for i in tqdm(range(z.shape[0])):
        ptBin = np.zeros(300)
        binValueMatrix = np.linspace(-14.9,15,300).round(2)
        ptValue = 0
        binValue = -14.9
        count = 0 
        tracksBin = np.zeros(300)
        trackCount = 0
        while binValue <= 15 and count < len(z[i]):

            if z[i][count] <= binValue:
                ptValue += (pt[i][count])
                count += 1
                trackCount += 1
            else:
                index = np.where(binValueMatrix == round(binValue,1))[0]
                ptBin[index] = ptValue
                binValue += 0.1
                ptValue = 0
                tracksBin[index] = trackCount
                trackCount = 0

        ptBinnedMatrix[i] = ptBin
        trackBinnedMatrix[i] = tracksBin
    print(trackBinnedMatrix[:2])
    print(ptBinnedMatrix[:2])
    return ptBinnedMatrix, trackBinnedMatrix

# name = "TTbar.root"
# n = "GTTObjects_ttbar200PU_1.root"
# eventZ,  eventPT, eventPV = loadData(name)

# zRaw, ptRaw, trackLength, mv = rawPaddedData(eventZ, eventPT)

# # print()
# # print(zRaw[0])
# # print(ptRaw[0])

# mv = np.array([mv])

# np.savez('TTbarRaw3', z=zRaw, pt=ptRaw, pv=np.array(eventPV), tl=trackLength, maxValue=mv)

rawD = np.load('TTbarRaw3.npz')
zRaw, ptRaw = rawD['z'], rawD['pt']
ptBin, trackBin = histogramData(zRaw, ptRaw)

np.savez('TTbarBin4', ptB=ptBin, tB=trackBin)

q = np.load('TTbarBin4.npz')
print(q['ptB'].shape)
print(q['tB'].shape)


