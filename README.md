# rail_phoebe

Photo-z BPZ Extension (phoebe): a fast/approximate version of BPZ that, rather than computing the full set of likelihoods, computes only a subset of nearest neighbor likelihoods in flux space.  This will give very similar PDFs to BPZ for bright/high S/N galaxies, but will show differences for low S/N data.

This code shares the bulk of its code with [rail_bpz](https://github.com/LSSTDESC/rail_bpz)  and also relies on [DESC_BPZ](https://github.com/LSSTDESC/DESC_BPZ).  Anyone using any version of BPZ via either rail_phoebe, rail_bpz, or DESC_BPZ should cite both [Benitez (2000)](https://ui.adsabs.harvard.edu/abs/2000ApJ...536..571B/abstract) and [Coe et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006AJ....132..926C/abstract).

See rail_bpz and DESC_BPZ for more documentation on the code.



