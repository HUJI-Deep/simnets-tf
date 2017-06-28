from .kmeans import kmeans_unsupervised_init as _kmeans_unsupervised_init
from .gmm import gmm_unsupervised_init as _gmm_unsupervised_init
from .pca import pca_unsupervised_init


def similarity_unsupervised_init(kind, sim_op, templates_var, weights_var):
    """Initialize a similarity layer using unsupervised learning

    Initializes the templates and weights using k-means or gmm (in k-means case the weights are just ones).
    The function returns two ops. The first is used to initialize the learning and the second should be run iteratively
     with all the data.

    Parameters
    ----------
    kind : {'kmeans', 'gmm'}
        type of unsupervised algorithm to use
    sim_op : tf.Operation | tf.Tensor
        the similarity operation (or the tensor which is the output of the similarity)
    templates_var : tf.Variable
        the templates variable for this similarity layer
    weights_var : tf.Variable
        the weights variable for this similarity layer

    Returns
    -------
    A tuple (init_op, update_op) where init_op must be executed by a session before using the update op
    and the update_op is the operation that performs the learning.
    """
    if kind == 'kmeans':
        return _kmeans_unsupervised_init(sim_op, templates_var, weights_var)
    elif kind == 'gmm':
        return _gmm_unsupervised_init(sim_op, templates_var, weights_var)
    else:
        raise ValueError('kind must be one of "kmeans" or "gmm"')