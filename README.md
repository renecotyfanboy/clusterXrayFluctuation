# X-ray surface brightness fluctuations code 

**[Link to internal doc](https://simon.dupourque.pages.in2p3.fr/fluctuation_xcop)**

This is the source code used to infer density fluctuations statistics from the
X-ray surface brightness fluctuations maps of galaxy clusters. It mostly enable the 
forward modelling of surface brightness maps using the GPU. It relies on both JAX and 
haiku libraries. It was used in the following paper:

- [Investigating the turbulent hot gas in X-COP galaxy clusters](https://ui.adsabs.harvard.edu/abs/2023arXiv230315102D/abstract)
- [CHEX-MATE: Turbulence in the intra-cluster medium from X-ray surface brightness fluctuations](https://ui.adsabs.harvard.edu/abs/2024A%26A...687A..58D/abstract)

## Installation

The sanest way to install the code is to create a fresh Python 3.10 conda environment

```bash
conda create -n cluster-xsb python=3.10
conda activate cluster-xsb
```

Then install the code using poetry

```bash
pip install poetry
poetry install
```
