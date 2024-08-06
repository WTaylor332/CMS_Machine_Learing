import numpy as np
import uproot
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm

import matplotlib.pyplot as plt

def loadData(dataFile):
    f = uproot.open(dataFile+':L1TrackNtuple/eventTree')
    events = f.arrays(['trk_pt', 'trk_eta','trk_phi','trk_z0','trk_nstub', 'trk_bendchi2','trk_chi2dof','pv_MC','pv_L1reco'])
    return events['trk_z0'], events['trk_pt'], events['pv_MC'], events['trk_eta']
  

def modifyData(z, pt, eta, ptCut=2):

    pt = np.where(pt<2, np.nan, pt)
    index = np.argwhere(np.isnan(pt))

    infIndex = np.argwhere(np.isinf(pt))
    pt[infIndex] = np.nan

    index = np.array(index).flatten()
    z[index] = np.nan
    z[infIndex] = np.nan

    index = np.array(index).flatten()
    eta[index] = np.nan
    eta[infIndex] = np.nan

    return z, pt, eta


def rawPaddedData(eventZ, eventPT, eventEta):
    padSize = 400
    eventZData = np.zeros((len(eventZ), padSize))
    eventPTData = np.zeros((len(eventPT), padSize))
    eventEtaData = np.zeros((len(eventEta), padSize))
    trackLength = np.zeros(len(eventZ))
    maxTLength = 0
    for i in tqdm(range(len(eventZ))):

        trackLength[i] = len(eventZ[i])

        zSort = np.zeros(padSize)
        padding = np.array([np.nan for x in range((padSize-len(eventZ[i])))])
        zSort[:len(eventZ[i])] = eventZ[i]
        zSort[len(eventZ[i]):] = padding


        ptSort = np.zeros(padSize)
        ptSort[:len(eventPT[i])] = eventPT[i]
        ptSort[len(eventPT[i]):] = padding

        etaSort = np.zeros(padSize)
        etaSort[:len(eventEta[i])] = eventEta[i]
        etaSort[len(eventEta[i]):] = padding

        z, pt, eta = modifyData(zSort, ptSort, etaSort)

        indexSort = np.argsort(z)
        zSort = np.sort(z)
        ptSort = pt[indexSort]
        etaSort = eta[indexSort]
        nanIndex = np.argwhere(np.isnan(zSort))
        if len(nanIndex) > 0:
            trackLength[i] = nanIndex[0][0]
            if nanIndex[0][0] > maxTLength:
                maxTLength = np.argwhere(np.isnan(zSort))[0][0]
        else:
            trackLength[i] = len(zSort)


        eventZData[i] = zSort
        eventPTData[i] = ptSort
        eventEtaData[i] = etaSort




    # to scale pt and eta
    zFlat = eventZData.flatten()
    ptFlat = eventPTData.flatten()
    etaFlat = eventEtaData.flatten()
    arr = np.stack((zFlat, ptFlat, etaFlat), axis=1)
    print(arr.shape)
    scaler = StandardScaler().fit(arr)
    arrScaled = scaler.transform(arr)
    eventPTData = arrScaled[:,1].reshape(len(eventPT), padSize)
    eventEtaData = arrScaled[:,2].reshape(len(eventEta), padSize)
    eventZData = eventZData[:, :maxTLength]
    eventPTData = eventPTData[:, :maxTLength]
    eventEtaData = eventEtaData[:, :maxTLength]
    print(eventZData.shape, eventPTData.shape, eventEtaData.shape)
    return eventZData, eventPTData, eventEtaData, trackLength, maxTLength

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
    return ptBinnedMatrix, trackBinnedMatrix


def merge():
    rawQ = np.load('QCD_Pt-15To3000.npz')
    rawW = np.load('WJetsToLNu.npz')
    rawT = np.load('TTbarRaw5.npz')

    print(rawQ['z'].shape, rawW['z'].shape, rawT['z'].shape)

    maxTrackLength = int(max([rawQ['maxValue'], rawT['maxValue'], rawW['maxValue']]))
    mergedEventsNo = int(rawQ['z'].shape[0] + rawW['z'].shape[0] + rawT['z'].shape[0])
    print(rawQ['z'].shape[1])
    print(maxTrackLength)
    padNoQ = int(maxTrackLength - rawQ['z'].shape[1])
    padNoW = int(maxTrackLength - rawW['z'].shape[1])
    padNoT = int(maxTrackLength - rawT['z'].shape[1])

    padQ = np.zeros((rawQ['z'].shape[0], padNoQ))
    padQ[padQ==0] = np.nan
    zQPadded = np.hstack((rawQ['z'], padQ))
    ptQPadded = np.hstack((rawQ['pt'], padQ))
    etaQPadded = np.hstack((rawQ['pt'], padQ))

    padW = np.zeros((rawW['z'].shape[0], padNoW))
    padW[padW==0] = np.nan
    zWPadded = np.hstack((rawW['z'], padW))
    ptWPadded = np.hstack((rawW['pt'], padW))
    etaWPadded = np.hstack((rawW['pt'], padW))

    padT = np.zeros((rawT['z'].shape[0], padNoT))
    padT[padT==0] = np.nan
    zTPadded = np.hstack((rawT['z'], padT))
    ptTPadded = np.hstack((rawT['pt'], padT))
    etaTPadded = np.hstack((rawT['pt'], padT))
    print(zTPadded[-1])
    print(rawT['z'][-1])
    print()

    zMerge = np.zeros((mergedEventsNo, maxTrackLength))
    ptMerge = np.zeros((mergedEventsNo, maxTrackLength))
    etaMerge = np.zeros((mergedEventsNo, maxTrackLength))

    zMerge[:rawQ['z'].shape[0]] = zQPadded = np.hstack((rawQ['z'], padQ))
    zMerge[rawQ['z'].shape[0]:rawQ['z'].shape[0]+rawW['z'].shape[0]] = zWPadded = np.hstack((rawW['z'], padW))
    zMerge[rawQ['z'].shape[0]+rawW['z'].shape[0]:] = zTPadded = np.hstack((rawT['z'], padT))
    print(zMerge[-1])
    print()
    ptMerge[:rawQ['z'].shape[0]] = ptQPadded = np.hstack((rawQ['pt'], padQ))
    ptMerge[rawQ['z'].shape[0]:rawQ['z'].shape[0]+rawW['z'].shape[0]] = ptWPadded = np.hstack((rawW['pt'], padW))
    ptMerge[rawQ['z'].shape[0]+rawW['z'].shape[0]:] = ptTPadded = np.hstack((rawT['pt'], padT))
    print(ptMerge[-1])
    print()
    etaMerge[:rawQ['z'].shape[0]] = etaQPadded = np.hstack((rawQ['pt'], padQ))
    etaMerge[rawQ['z'].shape[0]:rawQ['z'].shape[0]+rawW['z'].shape[0]] = etaWPadded = np.hstack((rawW['pt'], padW))
    etaMerge[rawQ['z'].shape[0]+rawW['z'].shape[0]:] = etaTPadded = np.hstack((rawT['pt'], padT))
    print(etaMerge[-1])
    print()

    print(zMerge.shape, ptMerge.shape, etaMerge.shape)
    print(zMerge[0])

    pv = np.concatenate((rawQ['pv'], rawW['pv'], rawT['pv']))
    trackLength = np.concatenate((rawQ['tl'], rawW['tl'], rawT['tl']))
    print(pv.shape)
    print(trackLength.shape)
    order = list(enumerate(pv))
    random.shuffle(order)
    shuffleIndex, newOrder = zip(*order)
    shuffleIndex =np.array(shuffleIndex)
    zMerge = zMerge[shuffleIndex]
    ptMerge = ptMerge[shuffleIndex]
    etaMerge = etaMerge[shuffleIndex]
    pvMerge = pv[shuffleIndex]
    trackLength = trackLength[shuffleIndex]

    print()
    print(zMerge.shape, ptMerge.shape, etaMerge.shape)
    print(zMerge[0])
    return zMerge, ptMerge, etaMerge, pvMerge, trackLength


def distributionTrack(z, bins):
    binValues = np.linspace(-15, 15+30/bins, bins)
    numTrackBin = np.zeros((z.shape[0], bins))
    for i in tqdm(range(z.shape[0])):
        for j in range(1, bins):
            rangeBin = z[i, (z[i]<binValues[j]) & (z[i]>binValues[j-1])]
            numTrackBin[i,j] = len(rangeBin)
    averageTrackEvent = np.mean(numTrackBin, axis=1)
    averageTrackBin = np.mean(numTrackBin, axis=0)
    print(np.amax(numTrackBin, axis=0))
    print(averageTrackBin)
    print(np.median(numTrackBin.flatten()))
    per = 99.9
    print(f'{per}th percentile: ', np.percentile(numTrackBin.flatten(), per))
    # fig, ax = plt.subplots(2, 2, figsize=(12,6), sharey=True)
    # ax[0, 0].hist(np.sort(z.flatten()), bins=30)
    # ax[0, 0].minorticks_on()
    # ax[0, 0].grid(which='both', alpha=0.7, c='#DDDDDD')
    # ax[0, 1].hist(np.sort(z.flatten()), bins=15)
    # ax[0, 1].minorticks_on()
    # ax[0, 1].grid(which='both', alpha=0.7, c='#DDDDDD')
    # ax[1, 0].hist(np.sort(numTrackBin.flatten()), bins=30)
    # ax[1, 0].minorticks_on()
    # ax[1, 0].grid(which='both', alpha=0.7, c='#DDDDDD')
    # ax[1, 1].hist(np.sort(numTrackBin.flatten()), bins=15)
    # ax[1, 1].minorticks_on()
    # ax[1, 1].grid(which='both', alpha=0.7, c='#DDDDDD')
    # plt.savefig('Track_distribution_for_each_bin.png', dpi=1000)
    return np.percentile(numTrackBin.flatten(), per)


def rawBinData(z, pt, eta, pv, binSize, per):
    maxTrackLength = per
    binValues = np.arange(-15, 15+binSize, binSize)
    print(binValues)
    zData = np.zeros((z.shape[0], len(binValues), maxTrackLength))
    ptData = np.zeros((z.shape[0], len(binValues), maxTrackLength))
    etaData = np.zeros((z.shape[0], len(binValues), maxTrackLength))
    pvData = np.zeros(z.shape[0]*len(binValues))
    hardVertexProb = np.zeros(z.shape[0]*len(binValues))
    for i in tqdm(range(z.shape[0])):
        for j in range(len(binValues)):
            zPad = np.zeros(maxTrackLength)
            ptPad = np.zeros(maxTrackLength)
            etaPad = np.zeros(maxTrackLength)
            valuesInBin = z[i, (z[i]<binValues[j]) & (z[i]>binValues[j-1])]
            if len(valuesInBin) > 0:
                index = np.argwhere((z[i]<binValues[j]) & (z[i]>binValues[j-1]))
                averageZ = np.nanmean(valuesInBin)
                if len(valuesInBin) > maxTrackLength:
                    zPad = valuesInBin[:maxTrackLength]
                    ptPad = pt[i, index[:maxTrackLength]].flatten()
                    etaPad = eta[i, index[:maxTrackLength]].flatten()
                else:
                    zPad[:len(valuesInBin)] = valuesInBin
                    zPad[len(valuesInBin):] = np.nan
                    ptPad[:len(valuesInBin)] = pt[i, index].flatten()
                    ptPad[len(valuesInBin):] = np.nan
                    etaPad [:len(valuesInBin)] = eta[i, index].flatten()
                    etaPad[len(valuesInBin):] = np.nan
                if pv[i] < binValues[j] and pv[i] > binValues[j-1]:
                    hardVertexProb[i*j] = 1
                    pvData[i*j] = pv[i]
                else:
                    hardVertexProb[i*j] = 0
                    pvData[i*j] = np.nan
                
                zData[i, j] = zPad
                ptData[i, j] = ptPad
                etaData[i, j] = etaPad
            else:
                averageZ = np.nan
                zPad[:] = np.nan
                ptPad[:] = np.nan
                etaPad[:] = np.nan
                zData[i, j] = zPad
                ptData[i, j] = ptPad
                etaData[i, j] = etaPad
    print(zData.shape, ptData.shape, etaData.shape)
    print(zData[0,4:10])

    return zData, ptData, etaData




# -------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------MAIN----------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------

# name = "TTbar.root"
# eventZ,  eventPT, eventPV, eventEta = loadData(name)

# zRaw, ptRaw, etaRaw, trackLength, mv = rawPaddedData(eventZ, eventPT, eventEta)
# np.savez('TTbarRaw5', z=zRaw, pt=ptRaw, eta=etaRaw, pv=np.array(eventPV), tl=trackLength, maxValue=np.array([mv]))

# rawD = np.load('TTbarRaw5.npz')
# zRaw, ptRaw, etaRaw = rawD['z'], rawD['pt'], rawD['eta']
# t = rawD['tl']
# m = rawD['maxValue']
# print(zRaw[0], ptRaw[0], etaRaw[0], '\n', t, '\n', m)

# ptBin, trackBin = histogramData(zRaw, ptRaw)

# np.savez('QCD_Pt-15To3000_Bin', ptB=ptBin, tB=trackBin)
# q = np.load('QCD_Pt-15To3000_Bin.npz')
# print(q['ptB'].shape)
# print(q['tB'].shape)


# merge and sort all decays

# zMerge, ptMerge, etaMerge, pvMerge, trackLength = merge()
# np.savez('Merged_deacys_Raw', z=zMerge, pt=ptMerge, eta=etaMerge, pv=np.array(pvMerge), tl=trackLength)
rawD = np.load('Merged_deacys_Raw.npz')
# z, pt, eta = mergeData['z'], mergeData['pt'], mergeData['eta']
# print()
# print(z.shape, pt.shape, eta.shape)

# # bin merged decays
# ptBin, trackBin = histogramData(z, pt)
# np.savez('Merged_decays_Bin', ptB=ptBin, tB=trackBin)

# q = np.load('Merged_decays_Bin.npz')
# print()
# print(q['ptB'].shape)
# print(q['tB'].shape)

# adding probability of hard vertext to mixed data
b = 30
# vertProb, vertBin = percentageHardVertex(mergeData['z'], mergeData['pv'], b)
# np.savez(f'Hard_Vertex_Probability_{b}_bins', prob=vertProb, bins=vertBin)

percentile = distributionTrack(rawD['z'], bins=b)

print()
binS = 30/b
zData, ptData, etaData = rawBinData(rawD['z'], rawD['pt'], rawD['eta'], rawD['pv'], binS, int(percentile))
np.savez(f'TTBar_Raw_{int(binS)}_bin_size', z=zData, pt=ptData, eta=etaData)

