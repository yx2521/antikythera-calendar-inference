import numpy as np
from scipy.stats import norm
import jax.numpy as jnp
from jax import grad

# ===================================================================
# ================== Q(b) Model the hole locations ==================

def phi(alphaj, i, N):
    """
    Compute the angular position φ_ij of the i-th hole in section j.

    Each hole is assumed to be uniformly spaced along a circular ring 
    with N total holes, and the angular offset of section j is alphaj.

    Parameters:
    alphaj (float or ndarray): Angular offset (in radians) of section j.
    i (int or ndarray): Index of the hole within the full ring (0-based).
    N (int): Total number of holes around the complete ring.

    Returns:
    float or ndarray: Angular position(s) φ_ij in radians.
    """
    
    return 2*np.pi*i/N + alphaj

def rij_unit(alphaj, i, N):
    """
    Computes the radial unit vectors for the hole at angle φ_ij.

    Parameters:
    alphaj (float or ndarray): Angular offset of section j (in radians).
    i (int or ndarray): Hole index.
    N (int): Total number of holes in the ring.

    Returns:
    tuple: (cos(φ_ij), sin(φ_ij)), the 2D radial unit vector.
    """

    phi_value = phi(alphaj, i, N)
    return np.cos(phi_value), np.sin(phi_value)

def tij_unit(alphaj, i, N):
    """
    Computes the tangential unit vector for the hole at angle φ_ij.

    Parameters:
    alphaj (float or ndarray): Angular offset of section j (in radians).
    i (int or ndarray): Hole index.
    N (int): Total number of holes in the ring.

    Returns:
    tuple: (sin(φ_ij), -cos(φ_ij)), the 2D tangential unit vector.
    """

    phi_value = phi(alphaj, i, N)
    return np.sin(phi_value), -np.cos(phi_value)


def rij_predict(xj, yj, r, phi):
    """
    Predict the Cartesian coordinates of a hole at angle φ_ij.

    Parameters:
    xj (float or ndarray): X-coordinate of the section's arc center.
    yj (float or ndarray): Y-coordinate of the section's arc center.
    r (float): Radius of the ring.
    phi (float or ndarray): Angular position φ_ij in radians.

    Returns:
    np.ndarray: Predicted hole position(s) [x, y], shape (2,) or (2, n).
    """

    rij_x = xj + r * np.cos(phi)
    rij_y = yj + r * np.sin(phi)
    return np.array([rij_x, rij_y])


def eij(datapoint, xj, yj, r, phi):
    """
    Compute the error vector e_ij, which represents the displacement 
    of the intended position in section j from the measured hole position.

    The error vector is given by:
        e_ij = r_ij - (d_i - r_0j)
    where:
    - r_ij is the theoretical position of the hole relative to its arc center 
      (≠ rij_predict which is defined relative to the origin).
    - d_i is the measured hole position.
    - r_0j = (xj, yj) is the arc center of section j.

    Parameters:
    datapoint (array): Measured hole position [x, y].
    xj (float): X coordinate of the section's arc center.
    yj (float): Y coordinate of the section's arc center.
    r (float): Radius of the ring.
    phi (float): Angular position of the i-th hole in section j in radians.

    Returns:
    numpy.ndarray: The error vector e_ij as a 2D array [ex, ey].
    """
    rij_value = rij_predict(xj, yj, r, phi)
    eij_value = rij_value - datapoint
    return eij_value


# =========================================================================
# ================== Q(c) Model the likelhiood functions ==================

# (1) *Model XY*: 2D isotropic model with one variance in x and y directions
def model_XY(sample, N, r, sigma_iso, 
             xj_1, xj_2, xj_3, xj_5, xj_6, xj_7,
             yj_1, yj_2, yj_3, yj_5, yj_6, yj_7,
             alphaj_1, alphaj_2, alphaj_3, alphaj_5, alphaj_6, alphaj_7):
    """
    Evaluate the isotropic 2D Gaussian likelihood for a set of measured hole positions.

    The model assumes independent and identically distributed Gaussian errors 
    in the Cartesian (X, Y) coordinates, with a common standard deviation sigma_iso.

    Parameters:
    sample (array): array([j], [i], [x], [y]), where:
        - j: Section IDs for each hole.
        - i: Hole indices.
        - x: Measured X coordinates.
        - y: Measured Y coordinates.
    N (int): Total number of holes in the complete ring.
    r (float): Radius of the calendar ring.
    sigma_iso (float): Standard deviation of isotropic error model.
    xj_k, yj_k, alphaj_k (float): Arc center (x, y) and rotation angle α for section k (k ∈ {1,2,3,5,6,7}).

    Returns:
    np.ndarray: Likelihood values under the isotropic Gaussian model for each sample point.
    """

    j, i, x, y = sample

    xj_dict = {1: xj_1, 2: xj_2, 3: xj_3, 5: xj_5, 6: xj_6, 7: xj_7}
    yj_dict = {1: yj_1, 2: yj_2, 3: yj_3, 5: yj_5, 6: yj_6, 7: yj_7}
    alphaj_dict = {1: alphaj_1, 2: alphaj_2, 3: alphaj_3, 5: alphaj_5, 6: alphaj_6, 7: alphaj_7}


    xj = np.array([xj_dict.get(a) for a in j])
    yj = np.array([yj_dict.get(a) for a in j])
    alphaj = np.array([alphaj_dict.get(a) for a in j])

    phi_ij = phi(alphaj, i, N)


    eij_vec = eij(np.array([x, y]), xj, yj, r, phi_ij)

    e_x = eij_vec[0 , :]
    e_y = eij_vec[1 , :]

    p_r = norm.pdf(e_x, loc=0, scale = sigma_iso)
    p_t = norm.pdf(e_y, loc=0, scale = sigma_iso)
    p = p_r * p_t

    return p


# (2) *Model RT*: 2D non-isotropic model with two vairiances in radial and tangential directions
def model_RT(sample, N, r, sigma_r, sigma_t, 
             xj_1, xj_2, xj_3, xj_5, xj_6, xj_7,
             yj_1, yj_2, yj_3, yj_5, yj_6, yj_7,
             alphaj_1, alphaj_2, alphaj_3, alphaj_5, alphaj_6, alphaj_7):
    """
    Evaluate the anisotropic 2D Gaussian likelihood with radial and tangential components.

    The model decomposes positional error into radial and tangential directions at each hole,
    and applies separate Gaussian distributions with standard deviations sigma_r and sigma_t.

    Parameters:
    sample (array): array([j], [i], [x], [y]), where:
        - j: Section IDs for each hole.
        - i: Hole indices.
        - x: Measured X coordinates.
        - y: Measured Y coordinates.
    N (int): Total number of holes in the complete ring.
    r (float): Radius of the calendar ring.
    sigma_r (float): Standard deviation of error in radial direction.
    sigma_t (float): Standard deviation of error in tangential direction.
    xj_k, yj_k, alphaj_k (float): Arc center (x, y) and rotation angle α for section k (k ∈ {1,2,3,5,6,7}).

    Returns:
    np.ndarray: Likelihood values under the non-isotropic Gaussian model for each sample point.
    """

    j, i, x, y = sample

    xj_dict = {1: xj_1, 2: xj_2, 3: xj_3, 5: xj_5, 6: xj_6, 7: xj_7}
    yj_dict = {1: yj_1, 2: yj_2, 3: yj_3, 5: yj_5, 6: yj_6, 7: yj_7}
    alphaj_dict = {1: alphaj_1, 2: alphaj_2, 3: alphaj_3, 5: alphaj_5, 6: alphaj_6, 7: alphaj_7}

    xj = np.array([xj_dict.get(a) for a in j])
    yj = np.array([yj_dict.get(a) for a in j])
    alphaj = np.array([alphaj_dict.get(a) for a in j])

    phi_ij = phi(alphaj, i, N)
    rij_unitvec = np.array(rij_unit(alphaj, i, N))
    tij_unitvec = np.array(tij_unit(alphaj, i, N))


    eij_vec = eij(np.array([x, y]), xj, yj, r, phi_ij)

    e_r = np.einsum("ij,ij->i", eij_vec.T, rij_unitvec.T)
    e_t = np.einsum("ij,ij->i", eij_vec.T, tij_unitvec.T)

    p_r = norm.pdf(e_r, loc=0, scale = sigma_r)
    p_t = norm.pdf(e_t, loc=0, scale = sigma_t)
    p = p_r * p_t

    return p


# ==================================================================================================================
# ================== Q(d) Convert the models into jax functions and model negative log-likelihood ==================

# (d)(1) Turn the functions into jax functions

def phi_jax(alphaj, i, N):
    """
    JAX version of `phi`. Computes the angular position φ_ij of the i-th hole in section j.
    Parameters/Returns are the jax-compatible versions of the same as in `phi`.
    """
    return 2*jnp.pi*i/N + alphaj

def rij_unit_jax(alphaj, i, N):
    """
    JAX version of `rij_unit`. Computes the radial unit vectors for the hole at angle φ_ij.
    Parameters/Returns are the jax-compatible versions of the same as in `rij_unit`.
    """
    phi_value = phi_jax(alphaj, i, N)
    return jnp.cos(phi_value), jnp.sin(phi_value)

def tij_unit_jax(alphaj, i, N):
    """
    JAX version of `tij_unit`. Computes the tangential unit vector for the hole at angle φ_ij.
    Parameters/Returns are the jax-compatible versions of the same as in `tij_unit`.
    """
    phi_value = phi_jax(alphaj, i, N)
    return jnp.sin(phi_value), -jnp.cos(phi_value)


def rij_predict_jax(xj, yj, r, phi):
    """
    JAX version of `rij_predict`. Computes the predicted hole position r_ij in Cartesian coordinates.
    Parameters/Returns are the jax-compatible versions of the same as in `rij_predict`.
    """

    rij_x = xj + r * jnp.cos(phi)
    rij_y = yj + r * jnp.sin(phi)
    return jnp.array([rij_x, rij_y])


def eij_jax(datapoint, xj, yj, r, phi):
    """
    JAX version of `eij`. Computes the error vector between predicted and measured hole positions.
    Parameters/Returns are the jax-compatible versions of the same as in `eij`.
    """
    rij_value = rij_predict_jax(xj, yj, r, phi)
    eij_value = rij_value - datapoint
    return eij_value


# (d)(2) Turn the likelihoods into jax functions

def normal_pdf(x, sigma):
    """
    JAX-compatible standard normal probability density function.

    Parameters:
    x (float or jnp.ndarray): Value(s) at which to evaluate the density.
    sigma (float): Standard deviation of the normal distribution.

    Returns:
    jnp.ndarray: Probability density values evaluated at x.
    """
    return (1.0 / (jnp.sqrt(2 * jnp.pi) * sigma)) * jnp.exp(-0.5 * (x / sigma)**2)



def model_XY_jax(sample, N, r, sigma_iso, 
             xj_1, xj_2, xj_3, xj_5, xj_6, xj_7,
             yj_1, yj_2, yj_3, yj_5, yj_6, yj_7,
             alphaj_1, alphaj_2, alphaj_3, alphaj_5, alphaj_6, alphaj_7):
    """
    JAX version of `model_XY`. Evaluates the isotropic 2D Gaussian likelihood 
    in Cartesian coordinates for hole positions."
    Parameters/Returns are the jax-compatible versions of the same as in `model_XY`.
    """

    j, i, x, y = sample


    xj_dict = {1: xj_1, 2: xj_2, 3: xj_3, 5: xj_5, 6: xj_6, 7: xj_7}
    yj_dict = {1: yj_1, 2: yj_2, 3: yj_3, 5: yj_5, 6: yj_6, 7: yj_7}
    alphaj_dict = {1: alphaj_1, 2: alphaj_2, 3: alphaj_3, 5: alphaj_5, 6: alphaj_6, 7: alphaj_7}

    xj = jnp.array([xj_dict.get(a) for a in j])
    yj = jnp.array([yj_dict.get(a) for a in j])
    alphaj = jnp.array([alphaj_dict.get(a) for a in j])

    phi_ij = phi_jax(alphaj, i, N)

    eij_vec = eij_jax(jnp.array([x, y]), xj, yj, r, phi_ij)

    e_x = eij_vec[0, :]
    e_y = eij_vec[1, :]

    p_r = normal_pdf(e_x, sigma_iso)
    p_t = normal_pdf(e_y, sigma_iso)
    p = p_r * p_t

    return p


def model_RT_jax(sample, N, r, sigma_r, sigma_t, 
             xj_1, xj_2, xj_3, xj_5, xj_6, xj_7,
             yj_1, yj_2, yj_3, yj_5, yj_6, yj_7,
             alphaj_1, alphaj_2, alphaj_3, alphaj_5, alphaj_6, alphaj_7):
    """
    JAX version of `model_RT`. Evaluates the non-isotropic 2D Gaussian likelihood 
    in the local radial/tangential frame of each hole."
    Parameters/Returns are the jax-compatible versions of the same as in `model_RT`.
    """
    
    j, i, x, y = sample


    xj_dict = {1: xj_1, 2: xj_2, 3: xj_3, 5: xj_5, 6: xj_6, 7: xj_7}
    yj_dict = {1: yj_1, 2: yj_2, 3: yj_3, 5: yj_5, 6: yj_6, 7: yj_7}
    alphaj_dict = {1: alphaj_1, 2: alphaj_2, 3: alphaj_3, 5: alphaj_5, 6: alphaj_6, 7: alphaj_7}

    xj = jnp.array([xj_dict.get(a) for a in j])
    yj = jnp.array([yj_dict.get(a) for a in j])
    alphaj = jnp.array([alphaj_dict.get(a) for a in j])

    phi_ij = phi_jax(alphaj, i, N)

    rij_unitvec = jnp.array(rij_unit_jax(alphaj, i, N))
    tij_unitvec = jnp.array(tij_unit_jax(alphaj, i, N))

    eij_vec = eij_jax(jnp.array([x, y]), xj, yj, r, phi_ij)

    e_r = jnp.einsum("ij,ij->i", eij_vec.T, rij_unitvec.T)
    e_t = jnp.einsum("ij,ij->i", eij_vec.T, tij_unitvec.T)

    p_r = normal_pdf(e_r, sigma_r)
    p_t = normal_pdf(e_t, sigma_t)
    p = p_r * p_t

    return p


# (d)(3) Define negative log-likelihood functions using jax

# Negative log-likelihood jax function for the isotropic model
def NLL_XY_for_jax(params, data):
    """
    JAX-compatible negative log-likelihood for the isotropic Gaussian error model.

    Parameters:
    params (tuple): Model parameters:
        - N (int): Total number of holes in the ring.
        - r (float): Ring radius.
        - sigma_iso (float): Standard deviation of isotropic Gaussian noise.
        - xj_k, yj_k, alphaj_k (float): Center coordinates and rotation angles for sections k ∈ {1,2,3,5,6,7}.
    data (array): array([j, i, x, y]) representing the hole observations.

    Returns:
    float: Total negative log-likelihood.
    """

    (N, r, sigma_iso,
     xj_1, xj_2, xj_3, xj_5, xj_6, xj_7,
     yj_1, yj_2, yj_3, yj_5, yj_6, yj_7,
     alphaj_1, alphaj_2, alphaj_3, alphaj_5, alphaj_6, alphaj_7) = params

    probs = model_XY_jax(
        data, N, r, sigma_iso,
        xj_1, xj_2, xj_3, xj_5, xj_6, xj_7,
        yj_1, yj_2, yj_3, yj_5, yj_6, yj_7,
        alphaj_1, alphaj_2, alphaj_3, alphaj_5, alphaj_6, alphaj_7
    )

    logp = jnp.log(probs + 1e-12)
    return -jnp.sum(logp) 


# Negative log-likelihood jax function for the non-isotropic model
def NLL_RT_for_jax(params, data):
    """
    JAX-compatible negative log-likelihood for the non-isotropic (radial/tangential) Gaussian error model.

    This function evaluates the total negative log-probability of the data 
    under the model assuming separate radial and tangential Gaussian noise components.

    Parameters:
    params (tuple): Model parameters:
        - N (int): Total number of holes in the ring.
        - r (float): Ring radius.
        - sigma_r (float): Std deviation of radial Gaussian noise.
        - sigma_t (float): Std deviation of tangential Gaussian noise.
        - xj_k, yj_k, alphaj_k (float): Center coordinates and rotation angles for sections k ∈ {1,2,3,5,6,7}.
    data (array): array([j, i, x, y]) representing the hole observations.

    Returns:
    float: Total negative log-likelihood.
    """

    (N, r, sigma_r, sigma_t,
     xj_1, xj_2, xj_3, xj_5, xj_6, xj_7,
     yj_1, yj_2, yj_3, yj_5, yj_6, yj_7,
     alphaj_1, alphaj_2, alphaj_3, alphaj_5, alphaj_6, alphaj_7) = params

    probs = model_RT_jax(
        data, N, r, sigma_r, sigma_t,
        xj_1, xj_2, xj_3, xj_5, xj_6, xj_7,
        yj_1, yj_2, yj_3, yj_5, yj_6, yj_7,
        alphaj_1, alphaj_2, alphaj_3, alphaj_5, alphaj_6, alphaj_7
    )

    logp = jnp.log(probs + 1e-12)
    return -jnp.sum(logp)




# =========================================================================
# ================== Q(e) Find MLE and plot preddictions ==================

# Calculate the predcited hole positions using the optimal parameters
def predict_hole_pos(filtered_data, params, model):
    """
    Predict the Cartesian positions of holes on the Antikythera mechanism plate,
    based on the model parameters and section indices.

    Parameters
    ----------
    filtered_data : array-like of shape (n_samples, 2)
        Each row contains [section_index (int), hole_index (float)] for a hole
        whose position is to be predicted.

    params : array-like of shape (n_params,)
        Model parameters. The interpretation depends on the selected model:
        
        - If model == "XY":
            [N, r, sigma_iso, xj_1, ..., xj_7, yj_1, ..., yj_7, alphaj_1, ..., alphaj_7]
            where sections 4 is omitted.
        
        - If model != "XY":
            [N, r, sigma_r, sigma_t, xj_1, ..., alphaj_7]

        Note: Only sections 1, 2, 3, 5, 6, 7 are used.

    model : str
        Model type, either "XY" for isotropic model or other (e.g. "RT") for anisotropic.
        Affects how parameters are unpacked.

    Returns
    -------
    np.ndarray of shape (n_samples, 4)
        Predicted hole positions. Each row is:
        [section_index (int), hole_index (float), x (float), y (float)]

    """
    if model == "XY":
        (N, r, sigma_iso,
         xj_1, xj_2, xj_3, xj_5, xj_6, xj_7,
         yj_1, yj_2, yj_3, yj_5, yj_6, yj_7,
         alphaj_1, alphaj_2, alphaj_3, alphaj_5, alphaj_6, alphaj_7) = params
    else:
        (N, r, sigma_r, sigma_t,
         xj_1, xj_2, xj_3, xj_5, xj_6, xj_7,
         yj_1, yj_2, yj_3, yj_5, yj_6, yj_7,
         alphaj_1, alphaj_2, alphaj_3, alphaj_5, alphaj_6, alphaj_7) = params
    
    section_params = {
        1: (xj_1, yj_1, alphaj_1),
        2: (xj_2, yj_2, alphaj_2),
        3: (xj_3, yj_3, alphaj_3),
        5: (xj_5, yj_5, alphaj_5),
        6: (xj_6, yj_6, alphaj_6),
        7: (xj_7, yj_7, alphaj_7),
    }


    predicted_full = []

    for row in filtered_data:
        j, i = int(row[0]), row[1]
        xj, yj, alphaj = section_params[j]
        phi_ij = phi(alphaj, i, N)
        predicted_position = rij_predict(xj, yj, r, phi_ij)
        predicted_full.append([j, i, predicted_position[0], predicted_position[1]])


    return np.array(predicted_full)



# =============================================================
# ================== Q(f) Posterior function ==================

# (f)(1) Define posterier distribution for the isotropic model
# Parameter Bounds
param_bounds_XY = jnp.array([
    [300, 400],     # N
    [0, 500],       # r
    [1e-8, 5.0],    # sigma_iso
    *[[0, 200]]*6,  # xj_1 to xj_7
    *[[0, 200]]*6,  # yj_1 to yj_7
    *[[-np.pi, np.pi]]*6  # alphaj_1 to alphaj_7
])

# Uniform prior log prob
def log_prior_XY(params):
    """
    Compute the log-prior for the XY model under uniform priors.

    This function assumes independent uniform priors for each parameter,
    defined by the bounds in `param_bounds_XY`. Since the log-density of a
    uniform distribution is constant within its support and zero elsewhere,
    this function returns:

        - 0.0 if all parameters are within bounds (i.e., log-const),
        - -inf if any parameter is out of bounds (i.e., log(0)).

    The constant log-density value (i.e., -log(b - a)) is omitted because it 
    cancels out in posterior ratios or MAP estimation, and thus has no impact 
    on inference.

    Parameters
    ----------
    params : jax.numpy.ndarray
        Parameter vector of shape (n_params,). Each parameter is assumed to have
        a uniform prior over the interval defined in `param_bounds_XY`.

    Returns
    -------
    float
        Log-prior value: 0.0 if all parameters are within bounds, -inf otherwise.
    """
    inside = jnp.all((params >= param_bounds_XY[:, 0]) & (params <= param_bounds_XY[:, 1]))
    return jnp.where(inside, 0.0, -jnp.inf)

# Posterior function for the isotropic model
def log_posterior_XY(params, data):
    """
    Compute the log-posterior for the XY model.

    Parameters
    ----------
    params : jax.numpy.ndarray
        Parameter vector of shape (n_params,).
    data : Any
        Observed data used in the likelihood function.

    Returns
    -------
    float
        Log-posterior = log-prior - negative log-likelihood.
        Returns -inf if parameters are out of bounds.
    """
    lp = log_prior_XY(params)
    return lp - NLL_XY_for_jax(params, data)


# (f)(2) Define posterier distribution for the non-isotropic model
# Parameter Bounds
param_bounds_RT = jnp.array([
    [300, 400],     # N
    [0, 500],       # r
    [1e-8, 5.0],    # sigma_r
    [1e-8, 5.0],    # sigma_t
    *[[0, 200]]*6,  # xj_1 to xj_7
    *[[0, 200]]*6,  # yj_1 to yj_7
    *[[-np.pi, np.pi]]*6  # alphaj_1 to alphaj_7
])

# Uniform prior log prob
def log_prior_RT(params):
    """
    Compute the log-prior for the RT model under uniform priors.

    This function assumes independent uniform priors for each parameter,
    defined by the bounds in `param_bounds_RT`. Since the log-density of a
    uniform distribution is constant within its support and zero outside,
    this function returns:

        - 0.0 if all parameters are within bounds (i.e., log of constant),
        - -inf if any parameter is out of bounds (i.e., log(0)).

    The constant log-density term (i.e., -log(b - a)) is omitted intentionally,
    as it does not affect posterior ratios, MCMC sampling, or MAP estimation.

    Parameters
    ----------
    params : jax.numpy.ndarray
        Parameter vector of shape (n_params,). Each parameter is assumed to have
        a uniform prior over the interval defined in `param_bounds_RT`.

    Returns
    -------
    float
        Log-prior value: 0.0 if all parameters are within bounds, -inf otherwise.
    """
    inside = jnp.all((params >= param_bounds_RT[:, 0]) & (params <= param_bounds_RT[:, 1]))
    return jnp.where(inside, 0.0, -jnp.inf)

def log_posterior_RT(params, data):
    """
    Compute the log-posterior for the RT model.

    Parameters
    ----------
    params : jax.numpy.ndarray
        Parameter vector of shape (n_params,).
    data : Any
        Observed data used in the likelihood function.

    Returns
    -------
    float
        Log-posterior = log-prior - negative log-likelihood.
        Returns -inf if parameters are out of bounds.
    """
    lp = log_prior_RT(params)
    return lp - NLL_RT_for_jax(params, data)


# (f)(3) Posterior predictive distribution for the isotropic model
def posterior_predictive_XY(HMC_samples, original_data):
    """
    Generate posterior predictive coordinates for a single input point using posterior samples.
    
    Parameters:
    - HMC_samples: Posterior samples (shape: n_samples, n_params)
    - original_data: True measured hole position
    
    Returns:
    - predictive_coordinates: Posterior predictive coordinates (shape: n_samples, 2)
    """
    j, i, x_orig, y_orig = original_data

    # Extract posterior parameters
    N = HMC_samples[:, 0]
    r = HMC_samples[:, 1]
    sigma_iso = HMC_samples[:, 2]
    xj_1, xj_2, xj_3, xj_5, xj_6, xj_7 = HMC_samples[:, 3], HMC_samples[:, 4], HMC_samples[:, 5], HMC_samples[:, 6], HMC_samples[:, 7], HMC_samples[:, 8]
    yj_1, yj_2, yj_3, yj_5, yj_6, yj_7 = HMC_samples[:, 9], HMC_samples[:, 10], HMC_samples[:, 11], HMC_samples[:, 12], HMC_samples[:, 13], HMC_samples[:, 14]
    alphaj_1, alphaj_2, alphaj_3, alphaj_5, alphaj_6, alphaj_7 = HMC_samples[:, 15], HMC_samples[:, 16], HMC_samples[:, 17], HMC_samples[:, 18], HMC_samples[:, 19], HMC_samples[:, 20]

    x = xj_3
    y = yj_3
    alpha = alphaj_3

    # Compute phi_ij for section 3
    phi_i = phi(alpha, i, N)

    rij = rij_predict(x, y, r, phi_i)

    # Sample new errors from the Gaussian distribution
    e_x_new = np.random.normal(loc=0, scale=sigma_iso)
    e_y_new = np.random.normal(loc=0, scale=sigma_iso)

    # Generate new coordinates
    x_new = rij[0] + e_x_new
    y_new = rij[1] + e_y_new

    # Combine new coordinates
    predictive_coordinates = np.column_stack((x_new, y_new))
    
    return predictive_coordinates

# (f)(4) Posterior predictive distribution for the non-isotropic model


def posterior_predictive_RT(samples, input_data):
    """
    Generate posterior predictive coordinates for a single input point using posterior samples
    for the non-isotropic 2D Gaussian model (Model RT).

    Parameters:
    - samples: Posterior samples (shape: n_samples, n_params)
    - input_data: True measured hole position

    Returns:
    - predictive_coordinates: Posterior predictive coordinates (shape: n_samples, 2)
    """
    # Unpack input data
    j, i, x_orig, y_orig = input_data

    # Extract posterior parameters
    N = samples[:, 0]
    r = samples[:, 1]
    sigma_r = samples[:, 2]
    sigma_t = samples[:, 3]
    xj_1, xj_2, xj_3, xj_5, xj_6, xj_7 = samples[:, 4], samples[:, 5], samples[:, 6], samples[:, 7], samples[:, 8], samples[:, 9]
    yj_1, yj_2, yj_3, yj_5, yj_6, yj_7 = samples[:, 10], samples[:, 11], samples[:, 12], samples[:, 13], samples[:, 14], samples[:, 15]
    alphaj_1, alphaj_2, alphaj_3, alphaj_5, alphaj_6, alphaj_7 = samples[:, 16], samples[:, 17], samples[:, 18], samples[:, 19], samples[:, 20], samples[:, 21]

    
    x = xj_3
    y = yj_3
    alpha = alphaj_3

    # Compute phi_ij for section 3
    phi_ij = phi(alpha, i, N)

    rij = rij_predict(x, y, r, phi_ij)
    radial_unit = rij_unit(alpha, i, N)
    tangential_unit = tij_unit(alpha, i, N)

    # Sample new errors from the Gaussian distributions
    e_r= np.random.normal(loc=0, scale=sigma_r, size=len(samples))  # Radial error
    e_t= np.random.normal(loc=0, scale=sigma_t, size=len(samples))  # Tangential error

    # Combine errors in radial and tangential directions
    e_x = e_r * radial_unit[0] + e_t * tangential_unit[0]
    e_y = e_r * radial_unit[1] + e_t * tangential_unit[1]

    # Generate new coordinates by adding errors to the predicted positions
    x_new = rij[0] + e_x
    y_new = rij[1] + e_y

    # Combine new coordinates
    predictive_coordinates = np.column_stack((x_new, y_new))
    
    return predictive_coordinates