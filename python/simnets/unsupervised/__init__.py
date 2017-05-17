from .kmeans import kmeans_unsupervised_init as _kmeans_unsupervised_init

def similarity_unsupervised_init(kind, sim_op, templates_var, weights_var):
    if kind == 'kmeans':
        return _kmeans_unsupervised_init(sim_op, templates_var, weights_var)
    elif kind == 'gmm':
        raise NotImplementedError('gmm is not implemeted yet')
    else:
        raise ValueError('kind must be one of "kmeans" or "gmm"')