import numpy as np
import os
import pytest
from rail.core.stage import RailStage
from rail.core.data import TableHandle
from rail.utils.path_utils import RAILDIR
from rail.utils.testing_utils import one_algo
from rail.estimation.algos import phoebe
from rail.phoebe.utils import RAIL_BPZ_DIR


parquetdata = "./tests/validation_10gal.pq"
fitsdata = "./tests/validation_10gal.fits"
traindata = os.path.join(RAILDIR, 'rail/examples_data/testdata/training_100gal.hdf5')
validdata = os.path.join(RAILDIR, 'rail/examples_data/testdata/validation_10gal.hdf5')

DS = RailStage.data_store
DS.__class__.allow_overwrite = True


@pytest.mark.parametrize(
    "ntarray, inputdata, groupname, size",
    [([8], parquetdata, "", 10),
     ([4, 4], traindata, "photometry", 100),
     ([8], fitsdata, "", 10),]
)
def test_bpz_train(ntarray, inputdata, groupname, size):
    # first, train with two broad types
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'dz': 0.01, 'hdf5_groupname': groupname,
                         'model': 'testmodel_bpz.pkl'}
    train_algo = phoebe.KNNBPZliteInformer
    DS.clear()
    training_data = DS.read_file('training_data', TableHandle, inputdata)
    train_stage = train_algo.make_stage(**train_config_dict)
    train_stage.inform(training_data)


def test_phoebe():
    train_config_dict = {}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0,
                         'dz': 0.01,
                         'nzbins': 301,
                         'data_path': None,
                         'spectra_file': "CWWSB4.list",
                         'madau_flag': 'no',
                         'no_prior': False,
                         'ref_band': 'mag_i_lsst',
                         'prior_file': 'hdfn_gen',
                         'p_min': 0.005,
                         'gauss_kernel': 0.0,
                         'zp_errors': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                         'mag_err_min': 0.005,
                         'hdf5_groupname': 'photometry',
                         'nt_array': [1, 2, 5],
                         'model': 'testmodel_bpz.pkl'}
    train_algo = None
    pz_algo = phoebe.KNNBPZliteEstimator
    results, rerun_results, rerun3_results = one_algo("BPZ_lite", train_algo, pz_algo, train_config_dict, estim_config_dict)
    # assert np.isclose(results.ancil['zmode'], zb_expected, atol=0.03).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()


def test_wrong_number_of_filters():
    train_config_dict = {}
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0,
                         'dz': 0.01,
                         'nzbins': 301,
                         'data_path': None,
                         'columns_file': os.path.join(RAIL_BPZ_DIR, "rail/examples_data/estimation_data/configs/test_bpz.columns"),
                         'spectra_file': "CWWSB4.list",
                         'madau_flag': 'no',
                         'ref_band': 'mag_i_lsst',
                         'prior_file': 'flat',
                         'p_min': 0.005,
                         'gauss_kernel': 0.1,
                         'zp_errors': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                         'mag_err_min': 0.005,
                         'filter_list': ['DC2LSST_u', 'DC2LSST_g'],
                         'hdf5_groupname': 'photometry'}
    train_algo = None
    with pytest.raises(ValueError):
        pz_algo = phoebe.KNNBPZliteEstimator
        _, _, _ = one_algo("BPZ_lite", train_algo, pz_algo, train_config_dict, estim_config_dict)
