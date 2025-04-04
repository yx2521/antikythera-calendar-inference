import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from all_code.hole_model import NLL_XY_for_jax, NLL_RT_for_jax

def load_data():
    file_path = "dataverse_files/1-Fragment_C_Hole_Measurements.csv"
    data = pd.read_csv(file_path)
    selected_columns = [data.columns[0], data.columns[1], data.columns[-2], data.columns[-1]]
    data_array = data[selected_columns].to_numpy()
    filtered_data = data_array[(data_array[:, 0] != 0) & (data_array[:, 0] != 4)]
    filtered_data_T = filtered_data.T
    return filtered_data_T

params_XY_test = jnp.array([
    360, 75, 0.5,
    80, 80, 80, 80, 80, 80,
    130, 130, 130, 130, 130, 130,
    -2.5, -2.5, -2.5, -2.5, -2.5, -2.5
], dtype=jnp.float32)

params_RT_test = jnp.array([
    360, 75, 0.5, 0.5,
    80, 80, 80, 80, 80, 80,
    130, 130, 130, 130, 130, 130,
    -2.5, -2.5, -2.5, -2.5, -2.5, -2.5
], dtype=jnp.float32)


def test_NLL_XY_grad_check():
    test_data = load_data()
    check_grads(lambda p: NLL_XY_for_jax(p, test_data),
                (params_XY_test,), order=1, modes=['fwd', 'rev'], atol=5e-3, rtol=5e-3)

def test_NLL_RT_grad_check():
    test_data = load_data()
    check_grads(lambda p: NLL_RT_for_jax(p, test_data),
                (params_RT_test,), order=1, modes=['fwd', 'rev'], atol=5e-3, rtol=5e-3)
