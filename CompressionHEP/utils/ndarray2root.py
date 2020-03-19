"""
This script saves a numpy ndarray (a detached torch tensor)
of single jet events (i.e. not jagged arrays)
back to a ROOT TTree, without ROOT or Athena.
TODO: Metadata?, compressiontypes
"""

import uproot
import numpy

#Specifies the 27D dataset. The available 'columns' can be read with ttree.keys()
prefix = 'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn'
branches4D = [
    # 4-momentum
    (prefix + '.pt',numpy.float64),
    (prefix + '.eta',numpy.float64),
    (prefix + '.phi',numpy.float64),
    (prefix + '.m',numpy.float64),
]


def ndarray_to_DxAOD(filename, array, branches=branches4D, compression=uproot.ZLIB):
    f = uproot.recreate(filename)
    
    branchdict = dict(branches)
    print(branchdict)
    
    f["CollectionTree"] = uproot.newtree(branchdict)
    #for i,branch  in enumerate(branches):
    #    data = array[:,i]
    #    print(branch[0])
    f["CollectionTree"].extend(dict([(branch[0],array[:,i]) for (i,branch) in enumerate(branches)]))