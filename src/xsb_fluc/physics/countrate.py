""" Python module containing structures to compute the countrate of the ICM using XSPEC."""
import os
import numpy as np
from abc import ABC, abstractmethod
from ..utils.config import config


if config.IS_TITAN:
    import xspec


class Countrate(ABC):

    def __init__(self):
        xspec.Xset.cosmo = "70, 0, 0.7"
        xspec.AllModels.setEnergies('0.5 1.4 1000 log')
        self.countrate = np.vectorize(self._countrate)

    @abstractmethod
    def _init_model(self):
        pass

    def _countrate(self, *pars, exp=1e6):

        xspec.AllData.clear()
        self.mod = self._init_model()

        fsPN = xspec.FakeitSettings(response=os.path.join(config.XMM_PATH, 'PN.rmf'),
                                    arf=os.path.join(config.XMM_PATH,'PN.arf'),
                                    exposure=0.6 * exp)

        fsM1 = xspec.FakeitSettings(response=os.path.join(config.XMM_PATH, 'M1.rmf'),
                                    arf=os.path.join(config.XMM_PATH, 'M1.arf'),exposure=0.2 * exp)

        fsM2 = xspec.FakeitSettings(response=os.path.join(config.XMM_PATH, 'M2.rmf'),
                                    arf=os.path.join(config.XMM_PATH, 'M2.arf'),
                                    exposure=0.2 * exp)
        fsPN.fileName = "PN.pha"
        fsM1.fileName = "M1.pha"
        fsM2.fileName = "M2.pha"

        fakeit_settings = [fsPN, fsM1, fsM2]

        self.mod.setPars(pars)
        xspec.AllData.fakeit(nSpectra=3, settings=fakeit_settings, applyStats=True, filePrefix="")
        xspec.AllData.ignore("0.0-0.7 1.2-**")

        rperPN = (np.array(xspec.AllData(1).rate))[3]
        rperM1 = (np.array(xspec.AllData(2).rate))[3]
        rperM2 = (np.array(xspec.AllData(3).rate))[3]

        return rperPN + rperM1 + rperM2  # counts s-1 cm5


class APEC(Countrate):
    """
     (kT, ab, z, Norm)
     """

    def _init_model(self):
        return xspec.Model('apec', modName='apec')


class PHabsAPEC(Countrate):
    """
    nH, kT, Ab, z, Norm
    """

    def _init_model(self):
        return xspec.Model('phabs*apec', modName='nhapec')