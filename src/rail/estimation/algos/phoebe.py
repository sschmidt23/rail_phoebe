"""
Port of *some* parts of BPZ, not the entire codebase.
Much of the code is directly ported from BPZ, written
by Txitxo Benitez and Dan Coe (Benitez 2000), which
was modified by Will Hartley and Sam Schmidt to make
it python3 compatible.  It was then modified to work
with TXPipe and ceci by Joe Zuntz and Sam Schmidt
for BPZPipe.  This version for RAIL removes a few
features and concentrates on just predicting the PDF.

Missing from full BPZ:
-no tracking of 'best' type/TB
-no "interp" between templates
-no ODDS, chi^2, ML quantities
-plotting utilities
-no output of 2D probs (maybe later add back in)
-no 'cluster' prior mods
-no 'ONLY_TYPE' mode

"""

import os
import numpy as np
# import pandas as pd
import scipy.integrate
import glob
import qp
import tables_io
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.utils.path_utils import RAILDIR
from rail.core.common_params import SHARED_PARAMS
from sklearn.neighbors import KDTree


default_prior_a = [2.465, 1.806, 0.906]
default_prior_fo = [0.35, 0.5]
default_prior_kt = [0.45, 0.147]
default_prior_km = [0.0913, 0.0636, 0.123]
default_prior_zo = [0.431, 0.39, 0.0626]
default_prior_nt = [1, 2, 5]
# prior_nt for cosmos136 is [7, 57, 72]


class KNNBPZliteInformer(CatInformer):
    """Inform stage for BPZliteEstimator, this stage *assumes* that you have a set of
    SED templates and that the training data has already been assigned a
    'best fit broad type' (that is, something like ellliptical, spiral,
    irregular, or starburst, similar to how the six SEDs in the CWW/SB set
    of Benitez (2000) are assigned 3 broad types).  This informer will then
    fit parameters for the evolving type fraction as a function of apparent
    magnitude in a reference band, P(T|m), as well as the redshift prior
    of finding a galaxy of the broad type at a particular redshift, p(z|m, T)
    where z is redshift, m is apparent magnitude in the reference band, and T
    is the 'broad type'.  We will use the same forms for these functions as
    parameterized in Benitez (2000).  For p(T|m) we have
    p(T|m) = exp(-kt(m-m0))
    where m0 is a constant and we fit for values of kt
    For p(z|T,m) we have

    ```
    P(z|T,m) = f_x*z0_x^a *exp(-(z/zm_x)^a)
    where zm_x = z0_x*(km_x-m0)
    ```

    where f_x is the type fraction from p(T|m), and we fit for values of
    z0, km, and a for each type.  These parameters are then fed to the BPZ
    prior for use in the estimation stage.
    """
    name = "KNNBPZliteInformer"
    config_options = CatInformer.config_options.copy()
    config_options.update(zmin=SHARED_PARAMS,
                          zmax=SHARED_PARAMS,
                          nzbins=SHARED_PARAMS,
                          bands=SHARED_PARAMS,
                          ref_band=SHARED_PARAMS,
                          filter_list=SHARED_PARAMS,
                          data_path=Param(str, "None",
                                          msg="data_path (str): file path to the "
                                          "SED, FILTER, and AB directories.  If left to "
                                          "default `None` it will use the install "
                                          "directory for rail + rail/examples_data/estimation_data/data"),
                          spectra_file=Param(str, "CWWSB4.list",
                                             msg="name of the file specifying the list of SEDs to use"),
                          leaf_size=Param(int, 15, msg="min leaf size for KDTree"),
                          madau_flag=Param(str, "no",
                                           msg="set to 'yes' or 'no' to set whether to include intergalactic "
                                               "Madau reddening when constructing model fluxes")
                          )

    def __init__(self, args, **kwargs):
        """Init function, init config stuff
        """
        super().__init__(args, **kwargs)
        
        self.mags = None
        self.szs = None
        datapath = self.config["data_path"]
        if datapath is None or datapath == "None":
            tmpdatapath = os.path.join(RAILDIR, "rail/examples_data/estimation_data/data")
            os.environ["BPZDATAPATH"] = tmpdatapath
            self.data_path = tmpdatapath
        else:  # pragma: no cover
            self.data_path = datapath
            os.environ["BPZDATAPATH"] = self.data_path
        if not os.path.exists(self.data_path):  # pragma: no cover
            raise FileNotFoundError("BPZDATAPATH " + self.data_path + " does not exist! Check value of data_path in config file!")


    def _initialize_run(self):
        super()._initialize_run()

        # If we are not the root process then we wait for
        # the root to (potentially) create all the templates before
        # reading them ourselves.
        if self.rank > 0:  # pragma: no cover
            # The Barrier method causes all processes to stop
            # until all the others have also reached the barrier.
            # If our rank is > 0 then we must be running under MPI.
            self.comm.Barrier()
            self.flux_templates = self._load_templates()
        # But if we are the root process then we just go
        # ahead and load them before getting to the Barrier,
        # which will allow the other processes to continue
        else:
            self.flux_templates = self._load_templates()
            # We might only be running in serial, so check.
            # If we are running MPI, then now we have created
            # the templates we let all the other processes that
            # stopped at the Barrier above continue and read them.
            if self.is_mpi():  # pragma: no cover
                self.comm.Barrier()

    def _make_new_ab_file(self, spectrum, filter_):  # pragma: no cover
        from desc_bpz.bpz_tools_py3 import ABflux

        new_file = f"{spectrum}.{filter_}.AB"
        print(f"  Generating new AB file {new_file}....")
        ABflux(spectrum, filter_, self.config.madau_flag)

    def _load_templates(self):
        from desc_bpz.useful_py3 import get_str, get_data, match_resol

        # The redshift range we will evaluate on
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        z = self.zgrid

        data_path = self.data_path
        filters = self.config.filter_list

        spectra_file = os.path.join(data_path, "SED", self.config.spectra_file)
        spectra = [s[:-4] for s in get_str(spectra_file)]

        nt = len(spectra)
        nf = len(filters)
        nz = len(z)
        zint = np.arange(nz)
        flux_templates = np.zeros((nz*nt, nf))
        template_redshifts = np.zeros(nz*nt, dtype=int)
        template_types = np.zeros(nz*nt, dtype=int)
        self.nt = nt
        self.nf = nf
        self.nz = nz
        
        ab_dir = os.path.join(data_path, "AB")
        os.makedirs(ab_dir, exist_ok=True)

        # make a list of all available AB files in the AB directory
        ab_file_list = glob.glob(ab_dir + "/*.AB")
        ab_file_db = [os.path.split(x)[-1] for x in ab_file_list]

        for i, s in enumerate(spectra):
            lowid = nz * i
            hiid = nz * (i+1)
            template_redshifts[lowid:hiid] = zint
            template_types[lowid:hiid] = np.repeat(i, nz)
            for j, f in enumerate(filters):
                model = f"{s}.{f}.AB"
                if model not in ab_file_db:  # pragma: no cover
                    self._make_new_ab_file(s, f)
                model_path = os.path.join(data_path, "AB", model)
                zo, f_mod_0 = get_data(model_path, (0, 1))
                flux_templates[lowid:hiid, j] = match_resol(zo, f_mod_0, z)

        return flux_templates, template_redshifts, template_types


    def run(self):
        """compute the best fit prior parameters
        """

        self.allfluxes, self.template_redshifts, self.template_types = self._load_templates()
        self.allcolors = np.zeros([self.nz*self.nt, self.nf])
        # find position of ref_band:
        iref = self.config.bands.index(self.config.ref_band)
        for i in range(self.nf):
            self.allcolors[:,i] = -2.5 * np.log10(self.allfluxes[:,i] / self.allfluxes[:,iref])

        # train kd-tree on colors
        colortree = KDTree(self.allcolors, leaf_size=self.config.leaf_size)

        self.model=dict(allfluxes=self.allfluxes, allcolors=self.allcolors, nz=self.nz, nt=self.nt,
                        nf=self.nf, colortree=colortree, template_redshifts=self.template_redshifts,
                        template_types=self.template_types, zmin=self.config.zmin,
                        zmax=self.config.zmax, nzbins=self.config.nzbins,
                        spectra_file=self.config.spectra_file)
        self.add_data("model", self.model)

        
class KNNBPZliteEstimator(CatEstimator):
    """CatEstimator subclass to implement basic marginalized PDF for BPZ
    In addition to the marginalized redshift PDF, we also compute several
    ancillary quantities that will be stored in the ensemble ancil data:
    zmode: mode of the PDF
    amean: mean of the PDF
    tb: integer specifying the best-fit SED *at the redshift mode*
    todds: fraction of marginalized posterior prob. of best template,
    so lower numbers mean other templates could be better fits, likely
    at other redshifts
    """
    name = "KNNBPZliteEstimator"
    config_options = CatEstimator.config_options.copy()
    config_options.update(nondetect_val=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          bands=SHARED_PARAMS,
                          ref_band=SHARED_PARAMS,
                          err_bands=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS,
                          zp_errors=SHARED_PARAMS,
                          filter_list=SHARED_PARAMS,
                          dz=Param(float, 0.01, msg="delta z in grid for Gauss convolution"),
                          data_path=Param(str, "None",
                                          msg="data_path (str): file path to the "
                                          "SED, FILTER, and AB directories.  If left to "
                                          "default `None` it will use the install "
                                          "directory for rail + ../examples_data/estimation_data/data"),
                          no_prior=Param(bool, False, msg="set to True if you want to run with no prior"),
                          p_min=Param(float, 0.0005,
                                      msg="BPZ sets all values of "
                                      "the PDF that are below p_min*peak_value to 0.0, "
                                      "p_min controls that fractional cutoff"),
                          gauss_kernel=Param(float, 0.0,
                                             msg="gauss_kernel (float): BPZ "
                                             "convolves the PDF with a kernel if this is set "
                                             "to a non-zero number"),
                          mag_err_min=Param(float, 0.005,
                                            msg="a minimum floor for the magnitude errors to prevent a "
                                            "large chi^2 for very very bright objects"),
                          prior_a=Param(list, default_prior_a, msg="alpha values for prior"),
                          prior_fo=Param(list, default_prior_fo, msg="fo values for prior"),
                          prior_kt=Param(list, default_prior_kt, msg="kt values for prior"),
                          prior_km=Param(list, default_prior_km, msg="km values for prior"),
                          prior_zo=Param(list, default_prior_zo, msg="zo values for prior"),
                          prior_mo=Param(float, 20.0, msg="m0 value for prior"),
                          prior_nt=Param(list, default_prior_nt, msg="nt values for prior"),
                          n_neigh=Param(int, 100, msg="number of neighbors kept in KD Tree"),
                          prior_grid_min=Param(float, 20.0, msg="value of lowest value of prior grid"),
                          prior_grid_max=Param(float, 25.5, msg="value of highest value for prior grid"),
                          prior_grid_dm=Param(float, 0.05, msg="delta mag for prior, grid"))

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do BPZ specific setup
        """
        super().__init__(args, **kwargs)

        datapath = self.config["data_path"]
        if datapath is None or datapath == "None":
            tmpdatapath = os.path.join(RAILDIR, "rail/examples_data/estimation_data/data")
            os.environ["BPZDATAPATH"] = tmpdatapath
            self.data_path = tmpdatapath
        else:  # pragma: no cover
            self.data_path = datapath
            os.environ["BPZDATAPATH"] = self.data_path
        if not os.path.exists(self.data_path):  # pragma: no cover
            raise FileNotFoundError("BPZDATAPATH " + self.data_path + " does not exist! Check value of data_path in config file!")

        # check on bands, errs, and prior band
        if len(self.config.bands) != len(self.config.err_bands):  # pragma: no cover
            raise ValueError("Number of bands specified in bands must be equal to number of mag errors specified in err_bands!")
        if self.config.ref_band not in self.config.bands:  # pragma: no cover
            raise ValueError(f"reference band not found in bands specified in bands: {str(self.config.bands)}")
        if len(self.config.bands) != len(self.config.err_bands) or len(self.config.bands) != len(self.config.filter_list):
            raise ValueError(
                f"length of bands {len(self.config.bands)}), "
                f"err_bands, {len(self.config.err_bands)} and "
                f"filter_list {len(self.config.filter_list)} are not the same!"
            )
        self.prior_dict = self._assemble_prior_dict()

    def _assemble_prior_dict(self):
        prior_dict = {}
        prior_dict['fo_arr'] = np.array(self.config.prior_fo)
        prior_dict['a_arr'] = np.array(self.config.prior_a)
        prior_dict['kt_arr'] = np.array(self.config.prior_kt)
        prior_dict['zo_arr'] = np.array(self.config.prior_zo)
        prior_dict['km_arr'] = np.array(self.config.prior_km)
        prior_dict['mo'] = self.config.prior_mo
        prior_dict['nt_array'] = np.array(self.config.prior_nt, dtype=int)
        return prior_dict

    def _make_prior_grid(self):
        from desc_bpz.prior_from_dict import prior_function
        self.prior_mgrid = np.arange(self.config.prior_grid_min, self.config.prior_grid_max+self.config.prior_grid_dm, self.config.prior_grid_dm)
        nm = len(self.prior_mgrid)
        ntot = np.sum(self.config.prior_nt)
        self.prior_grid = np.zeros([nm, self.nz, self.nt])
        for ii, mm in enumerate(self.prior_mgrid):
            self.prior_grid[ii] = prior_function(self.zgrid, mm, self.prior_dict, ntot)
    
    def _initialize_run(self):
        super()._initialize_run()

        # If we are not the root process then we wait for
        # the root to (potentially) create all the templates before
        # reading them ourselves.
        if self.rank > 0:  # pragma: no cover
            # The Barrier method causes all processes to stop
            # until all the others have also reached the barrier.
            # If our rank is > 0 then we must be running under MPI.
            self.comm.Barrier()
            self.flux_templates = self.allfluxes
        # But if we are the root process then we just go
        # ahead and load them before getting to the Barrier,
        # which will allow the other processes to continue
        else:
            self.flux_templates = self.allfluxes
            # We might only be running in serial, so check.
            # If we are running MPI, then now we have created
            # the templates we let all the other processes that
            # stopped at the Barrier above continue and read them.
            if self.is_mpi():  # pragma: no cover
                self.comm.Barrier()

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        self.modeldict = self.model
        self.allfluxes = self.model['allfluxes']
        self.allcolors = self.model['allcolors']
        self.nz = self.model['nz']
        self.nt = self.model['nt']
        self.nf = self.model['nf']
        self.colortree = self.model['colortree']
        self.template_redshifts = self.model['template_redshifts']
        self.template_types = self.model['template_types']
        self.zmin = self.model['zmin']
        self.zmax = self.model['zmax']
        self.nzbins = self.model['nzbins']
        self.zgrid = np.linspace(self.zmin, self.zmax, self.nzbins)

    def _preprocess_magnitudes(self, data):
        from desc_bpz.bpz_tools_py3 import e_mag2frac

        bands = self.config.bands
        errs = self.config.err_bands

        fluxdict = {}
        nchunk = len(data[self.config.ref_band])
        chunkcolors = np.zeros([nchunk, self.nf])
        
        # Load the magnitudes
        zp_frac = e_mag2frac(np.array(self.config.zp_errors))

        # replace non-detects with 99 and mag_err with lim_mag for consistency
        # with typical BPZ performance
        for ii, (bandname, errname) in enumerate(zip(bands, errs)):
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                detmask = np.isnan(data[bandname])
            else:
                detmask = np.isclose(data[bandname], self.config.nondetect_val)
            
            data[bandname][detmask] = self.config.mag_limits[bandname]
            data[errname][detmask] = self.config.mag_limits[bandname]
            chunkcolors[:, ii] = data[bandname] - data[self.config.ref_band]
            data[bandname][detmask] = 99.0

        # Only one set of mag errors
        mag_errs = np.array([data[er] for er in errs]).T

        # Group the magnitudes and errors into one big array
        mags = np.array([data[b] for b in bands]).T

        # Clip to min mag errors.
        # JZ: Changed the max value here to 20 as values in the lensfit
        # catalog of ~ 200 were causing underflows below that turned into
        # zero errors on the fluxes and then nans in the output
        np.clip(mag_errs, self.config.mag_err_min, 20, mag_errs)

        # Convert to pseudo-fluxes
        flux = 10.0**(-0.4 * mags)
        flux_err = flux * (10.0**(0.4 * mag_errs) - 1.0)

        # Check if an object is seen in each band at all.
        # Fluxes not seen at all are listed as infinity in the input,
        # so will come out as zero flux and zero flux_err.
        # Check which is which here, to use with the ZP errors below
        seen1 = (flux > 0) & (flux_err > 0)
        seen = np.where(seen1)
        # unseen = np.where(~seen1)
        # replace Joe's definition with more standard BPZ style
        nondetect = 99.
        nondetflux = 10.**(-0.4 * nondetect)
        unseen = np.isclose(flux, nondetflux, atol=nondetflux * 0.5)

        # replace mag = 99 values with 0 flux and 1 sigma limiting magnitude
        # value, which is stored in the mag_errs column for non-detects
        # NOTE: We should check that this same convention will be used in
        # LSST, or change how we handle non-detects here!
        flux[unseen] = 0.
        flux_err[unseen] = 10.**(-0.4 * np.abs(mag_errs[unseen]))

        # Add zero point magnitude errors.
        # In the case that the object is detected, this
        # correction depends onthe flux.  If it is not detected
        # then BPZ uses half the errors instead
        add_err = np.zeros_like(flux_err)
        add_err[seen] = ((zp_frac * flux)**2)[seen]
        add_err[unseen] = ((zp_frac * 0.5 * flux_err)**2)[unseen]
        flux_err = np.sqrt(flux_err**2 + add_err)

        # Upate the flux dictionary with new things we have calculated
        fluxdict['flux'] = flux
        fluxdict['flux_err'] = flux_err
        m_0_col = self.config.bands.index(self.config.ref_band)
        fluxdict['mag0'] = mags[:, m_0_col]

        return fluxdict, chunkcolors

    def _calculate_likelihood(self, f, ef, ft_z, t_szs, t_types):
        # and adaptation of the bits of p_c_z_t from bpz_tools_py3 that do the likelihood
        eps=1e-300
        eeps=np.log(eps)
        xnz, xnt, xnf = ft_z.shape
        foo=np.add.reduce(np.where(np.less(f/ef,1e-4),0.,(f/ef)**2))
        nonobs=np.greater(np.reshape(ef, (1, 1, xnf)) + ft_z*0., 1.0)
        fot=np.add.reduce(np.where(nonobs,0.,np.reshape(f, (1, self.nf)) * ft_z / np.reshape(ef, (1, self.nf))**2), -1)
        ftt=np.add.reduce(np.where(nonobs,0.,ft_z**2 / np.reshape(ef, (1, self.nf))**2), -1)
        chi2=np.where(np.equal(ftt,0.), foo, foo-(fot**2)/(ftt+eps)).flatten()
        # we now have a 1D array, don't need this search
        #chi2_minima=np.loc2d(chi2[:xnz,:xnt],'min')
        chi2_minima_pos = np.argmin(chi2)
        # i_z_ml=int(chi2_minima_pos)
        # i_t_ml=int(chi2_minima)
        # min_chi2=self.chi2[self.i_z_ml,self.i_t_ml]
        min_chi2 = chi2[chi2_minima_pos]
        likelihood = np.exp(-0.5*np.clip((chi2-min_chi2),0.,-2*eeps))
        # this is just a set of likelihoods for NN points, need to put on a grid
        likegrid = np.zeros([self.nz, self.nt])
        for ii, like in enumerate(likelihood):
            likegrid[t_szs[ii], t_types[ii]] = like
        return likegrid

    def _estimate_pdf(self, flux_templates, kernel, flux, flux_err, galcolor, mo):
        #from desc_bpz.prior_from_dict import prior_function

        eps=1e-300
        #priordict = self._assemble_prior_dict()
        
        #modeldict = self.modeldict
        p_min = self.config.p_min
        nt = flux_templates.shape[1]

        # use kdtree to find closest neighbors!
        dists, idxs = self.colortree.query(np.atleast_2d(galcolor), k=self.config.n_neigh)
        # pull out subsets of model fluxes that are matches
        knnfluxes = flux_templates[idxs]
        knnredshifts = self.template_redshifts[idxs].ravel()
        knntypes = self.template_types[idxs].ravel()
        
        # The likelihood and prior...
        #pczt = p_c_z_t(flux, flux_err, flux_templates)
        #L = pczt.likelihood

        # New likelihood calculation
        L = self._calculate_likelihood(flux, flux_err, knnfluxes, knnredshifts, knntypes)

        ## old prior code returns NoneType for prior if "flat" or "none"
        ## just hard code the no prior case for now for backward compatibility
        #if self.config.no_prior:  # pragma: no cover
        #    P = np.ones(L.shape)
        #else:
        #    # set num templates to nt, which is hardcoding to "interp=0"
        #    # in BPZ, i.e. do not create any interpolated templates
        #    P = prior_function(self.zgrid, mo, self.prior_dict, nt)
        if mo > self.config.prior_grid_max:  # pragma: no cover
            mo = self.prior_grid_max
        elif mo < self.config.prior_grid_min:  # pragma: no cover
            mo = self.config.prior_grid_min
        gridm = np.searchsorted(self.prior_mgrid, mo)
        P = self.prior_grid[gridm]
        
        post = L * P
        # Right now we jave the joint PDF of p(z,template). Marginalize
        # over the templates to just get p(z)
        post_z = post.sum(axis=1)

        # Convolve with Gaussian kernel, if present
        if kernel is not None:  # pragma: no cover
            post_z = np.convolve(post_z, kernel, 1)

        # Find the mode
        zpos = np.argmax(post_z)
        zmode = self.zgrid[zpos]

        # Trim probabilities
        # below a certain threshold pct of p_max
        p_max = post_z.max()
        post_z[post_z < (p_max * p_min)] = 0

        # Normalize in the same way that BPZ does
        # if all zero, then set to uniform distribution
        if np.isclose(post_z.sum(), 0.0):  # pragma: no cover
            post_z = np.ones(len(self.zgrid))
        else:
            post_z /= post_z.sum()

        # Find T_B, the highest probability template *at zmode*
        tmode = post[zpos, :]
        t_b = np.argmax(tmode)

        # compute TODDS, the fraction of probability of the "best" template
        # relative to the other templates
        tmarg = post.sum(axis=0)
        todds = tmarg[t_b] / (np.sum(tmarg) + eps)

        return post_z, zmode, t_b, todds

    def _process_chunk(self, start, end, data, first):
        """
        Run BPZ on a chunk of data
        """
        self._make_prior_grid()

        # replace non-detects, traditional BPZ had nondet=99 and err = maglim
        # put in that format here
        test_data, test_colors = self._preprocess_magnitudes(data)
        m_0_col = self.config.bands.index(self.config.ref_band)

        nz = self.nz
        ng = test_data['flux'].shape[0]

        # Set up Gauss kernel for extra smoothing, if needed
        if self.config.gauss_kernel > 0:  # pragma: no cover
            dz = self.config.dz
            x = np.arange(-3. * self.config.gauss_kernel,
                          3. * self.config.gauss_kernel + dz / 10., dz)
            kernel = np.exp(-(x / self.config.gauss_kernel)**2)
        else:
            kernel = None

        pdfs = np.zeros((ng, nz))
        zmode = np.zeros(ng)
        zmean = np.zeros(ng)
        tb = np.zeros(ng)
        todds = np.zeros(ng)
        flux_temps = self.allfluxes
        zgrid = self.zgrid
        # Loop over all ng galaxies!
        for i in range(ng):
            mag_0 = test_data['mag0'][i]
            flux = test_data['flux'][i]
            flux_err = test_data['flux_err'][i]
            galcolor = test_colors[i]
            pdfs[i], zmode[i], tb[i], todds[i] = self._estimate_pdf(flux_temps,
                                                                    kernel, flux,
                                                                    flux_err,
                                                                    galcolor, mag_0)
            zmean[i] = (zgrid * pdfs[i]).sum() / pdfs[i].sum()
        qp_dstn = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=pdfs))
        qp_dstn.set_ancil(dict(zmode=zmode, zmean=zmean, tb=tb, todds=todds))
        self._do_chunk_output(qp_dstn, start, end, first, data=data)
