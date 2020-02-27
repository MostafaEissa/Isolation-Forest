import numpy as np

class IsolationForest:
    
    def fit(self, xTr, m, sub_sampling_size=np.inf):
        n, d = xTr.shape
        trees = []
        indices = np.arange(n)
        if sub_sampling_size == np.inf:
            sub_sampling_size = n
        
        maxdepth = np.ceil(np.log(sub_sampling_size))
            
        for i in range(m):
            idx = np.random.choice(indices, sub_sampling_size, replace=True)
            t = _tree(xTr[idx,:],depth=maxdepth)
            trees.append(t)
        return trees
        
def _tree(xTr,depth=np.inf):
    n,d = xTr.shape
    
    if depth <= 1 or len(xTr) == 0 or (xTr == xTr[0]).all():
            return _TreeNode(None, None, None, None, None)
        
    fid,cut = _split(xTr)
    L_idx = xTr[:,fid] <= cut
    R_idx = xTr[:,fid] > cut
    tree_L = _tree(xTr[L_idx, :], depth-1)
    tree_R = _tree(xTr[R_idx, :], depth-1)
    root = _TreeNode(tree_L, tree_R, None, fid, cut)
    tree_L.parent = root
    tree_R.parent = root
    
    return root
 
def _split(xTr):
    N,D = xTr.shape
    assert D > 0 # must have at least one dimension
    assert N > 1 # must have at least two samples
    feature = np.inf
    cut = np.inf

    # randomly select an attribute
    feature = np.random.randint(D)
    
    # randomly select a cutoff point from max and min values of attribute
    f_max = np.max(xTr[:, feature])
    f_min = np.max(xTr[:, feature])
    cut = (f_max - f_min) *np.random.random_sample() + f_min
             
    return feature, cut

class _TreeNode(object):
    def __init__(self, left, right, parent, cutoff_id, cutoff_val):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
