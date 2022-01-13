# template for Sphinx compatible function
import numpy as np

# from nma_datasets 
#  show_images
# calculate_torch_RSM, calculate_numpy_RSM, plot_dsprites_RSMs
def is_classifier(estimator):
    """Return True if the given estimator is (probably) a classifier.
    Parameters
    ----------
    estimator : object
        Estimator object to test.
    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "classifier"

    public static void SphericalToCartesian(float radius, float polar, float elevation, out Vector3 outCart){
        float a = radius * Mathf.Cos(elevation);
        outCart.x = a * Mathf.Cos(polar);
        outCart.y = radius * Mathf.Sin(elevation);
        outCart.z = a * Mathf.Sin(polar);
    }

def spherical2cartesian(d, a, e):
    # paste from the implemented version
    tmp = d*np.cos(e)
    x = tmp* np.cos(a)
    y = d* np.sin(e)
    z = tmp*np.sin(a)
    return x,y,z