import haiku as hk
from .density import CleanVikhlininModel
from .temperature import GhirardiniModel
from .cooling import CoolingFunctionModel, CoolingFunctionGrid
from ..data.cluster import Cluster


class XrayEmissivity(hk.Module):
    """
    3D Xray emissivity build with temperature, cooling function, density model.
    It depends on the redshift of the cluster, since the cooling function is precomputed using XSPEC.
    The default models are the ones used in the papers i.e. Vikhlinin for density, Ghirardini for temperature
    and the interpolated cooling function.
    """
    def __init__(self, redshift):
        super(XrayEmissivity, self).__init__()
        self.squared_density = CleanVikhlininModel()
        self.temperature = GhirardiniModel()
        self.cooling_function = CoolingFunctionModel(CoolingFunctionGrid(n_points=10, redshift=redshift))

    @classmethod
    def from_data(cls, data: "Cluster"):
        """
        Create an emissivity model from a `Cluster` object
        """
        return cls(data.z)

    def __call__(self, r, nH):
        """
        Compute the emissivity at a given radius, including $N_H$ absorption.

        Parameters:
            r (jnp.array): radius in units of $R_{500}$
            nH (jnp.array): Hydrogen density in cm$^{-2}$ in the line of sight corresponding to the radius r
        """
        return self.cooling_function(nH, self.temperature(r)) * self.squared_density(r)
