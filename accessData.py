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
    print(averageTrackBin)
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


def rawBinData(z, pt, eta, pv, binSize, per, lap=0):
    maxTrackLength = per
    binValues = np.arange(-15, 15+lap, binSize-lap)
    noBins = int((30 / binSize * (binSize*lap))-1)
    if lap != 0:
        offsetValues = np.arange(-15+binSize, 15+binSize, binSize-lap)
        binValues = binValues[binValues<=15+lap]
        offsetValues = offsetValues[offsetValues<=15+lap]
        joinedValues = np.zeros((len(binValues)+len(offsetValues)))
        joinedValues[0::2] = binValues
        joinedValues[1::2] = offsetValues
        if len(joinedValues) % 2 != 0:
            joinedValues = joinedValues[:-3]
        binValues = joinedValues
    zData = np.zeros((z.shape[0], noBins, maxTrackLength))
    ptData = np.zeros((z.shape[0], noBins, maxTrackLength))
    etaData = np.zeros((z.shape[0], noBins, maxTrackLength))
    pvData = np.zeros((z.shape[0], noBins), dtype=float)
    pvData[pvData == 0] = np.nan
    hardVertexProb = np.zeros((z.shape[0], noBins))
    count = 0
    countPV = 0
    for i in tqdm(range(z.shape[0])):
        whichBin = 0
        for j in range(len(binValues)):
            zPad = np.zeros(maxTrackLength)
            ptPad = np.zeros(maxTrackLength)
            etaPad = np.zeros(maxTrackLength)
            valuesInBin = z[i, (z[i]<binValues[j]) & (z[i]>binValues[j-1])]
            if len(valuesInBin) > 0:
                index = np.argwhere((z[i]<binValues[j]) & (z[i]>binValues[j-1]))
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
                zData[i, whichBin] = zPad
                ptData[i, whichBin] = ptPad
                etaData[i, whichBin] = etaPad
            else:
                zPad[:] = np.nan
                ptPad[:] = np.nan
                etaPad[:] = np.nan
                zData[i, whichBin] = zPad
                ptData[i, whichBin] = ptPad
                etaData[i, whichBin] = etaPad
            
            if pv[i] < binValues[j] and pv[i] > binValues[j-1]:
                oddEven = j % 2
                if (pv[i] - binValues[j]) < 0 and (pv[i] - binValues[j-2]) > 0:
                    if abs(pv[i] - binValues[j]) > abs(pv[i] - binValues[j-2]):
                        hardVertexProb[i, whichBin] = 1
                        pvData[i, whichBin] = pv[i]
                        count += 1
                    else:
                        if i  > 2488 and i < 2492:
                            print(pv[i])
                            print(whichBin)
                            print(j)
                            print((pv[i] - binValues[j]), (pv[i] - binValues[j-2]))
                        if whichBin < 29:
                            hardVertexProb[i, whichBin+1] = 1
                            pvData[i, whichBin+1] = pv[i]
                            count += 1
                        else:
                            hardVertexProb[i, whichBin] = 1
                            pvData[i, whichBin] = pv[i]
                            count += 1
                
            whichBin = j // 2 
        
        if pv[i] > binValues[-1] or pv[i] < binValues[0]:
            countPV += 1

    pvData = pvData.flatten()
    hardVertexProb = hardVertexProb.flatten()
    print(countPV)
    print(count)
    print(zData.shape, ptData.shape, etaData.shape, pvData.shape, hardVertexProb.shape)

    return zData, ptData, etaData, pvData, hardVertexProb




# -------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------MAIN----------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------

# name = "TTbar.root"
# eventZ,  eventPT, eventPV, eventEta = loadData(name)

# zRaw, ptRaw, etaRaw, trackLength, mv = rawPaddedData(eventZ, eventPT, eventEta)
# np.savez('TTbarRaw5', z=zRaw, pt=ptRaw, eta=etaRaw, pv=np.array(eventPV), tl=trackLength, maxValue=np.array([mv]))

rawD = np.load('TTbarRaw5.npz')
nameData = 'TTbar'
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
# rawD = np.load('Merged_deacys_Raw.npz')
# nameData = 'Merged'
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

# adding probability of hard vertex to mixed data
b = 15

# percentile = distributionTrack(rawD['z'], bins=b)

print()
binS = 30/b
overlap = binS/2
print(nameData, binS)
print(rawD['pv'][:5])
# print(rawD['z'][0])
zData, ptData, etaData, pvData, probability = rawBinData(rawD['z'], rawD['pt'], rawD['eta'], rawD['pv'], binS, int(85), lap=overlap)
print(np.sum(probability))
print(probability.shape)
print(pvData.shape)
pv = pvData[~np.isnan(pvData)]
print(pv.shape)
np.savez(f'{nameData}_Raw_{int(binS)}_bin_size_overlap_pv_far_from_boundary_{overlap}', z=zData, pt=ptData, eta=etaData, pv=pvData, prob=probability)

