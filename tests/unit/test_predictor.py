"""Unit tests for PowerSystemInjectionPredictor."""

import numpy as np
from oedisi.types.data_types import Injection, Topology, VoltagesMagnitude

from imputation_federate.predictor import PowerSystemInjectionPredictor


def test_predictor_topology_init(mock_topology: Topology):
    """Test predictor initialization from OEDISI Topology."""
    predictor = PowerSystemInjectionPredictor(n_buses=132, n_phases=3)
    predictor.init_from_topology(mock_topology)

    assert predictor.n_buses == 132
    assert len(predictor.bus_names) == 132
    assert predictor.base_r is not None
    assert predictor.base_x is not None


def test_predictor_tensors_extraction(
    mock_topology: Topology,
    mock_voltages_magnitude: VoltagesMagnitude,
    mock_injections: Injection,
):
    """Test conversion of OEDISI data types to numpy tensors."""
    predictor = PowerSystemInjectionPredictor(n_buses=132, n_phases=3)
    predictor.init_from_topology(mock_topology)

    v_tensor = predictor.extract_voltage_tensor(mock_voltages_magnitude)
    assert v_tensor.shape == (1, 132, 3)
    assert np.any(v_tensor > 0)

    p_tensor = predictor.extract_power_tensor(powers_real=None, injections=mock_injections)
    assert p_tensor.shape == (1, 132, 3)
    assert np.any(p_tensor > 0)


def test_predictor_synthetic_training_and_inference(mock_topology: Topology):
    """Test model building, training, and prediction cycle."""
    n_samples = 4
    n_buses = 132
    n_phases = 3
    n_branches = 131

    v_data = np.random.normal(2400.0, 10.0, (n_samples, n_buses, n_phases))
    l_data = np.random.normal(10.0, 2.0, (n_samples, n_buses, n_phases))
    r_data = np.full((n_samples, n_branches, n_phases, n_phases), 0.05)
    x_data = np.full((n_samples, n_branches, n_phases, n_phases), 0.15)
    inj_data = np.random.normal(50.0, 5.0, (n_samples, n_buses, n_phases))

    predictor = PowerSystemInjectionPredictor(n_buses=n_buses, n_phases=n_phases)
    predictor.init_from_topology(mock_topology)

    history = predictor.train(v_data, l_data, r_data, x_data, inj_data, epochs=1, batch_size=4)
    assert "loss" in history.history

    pred_inj = predictor.predict(v_data[:1], l_data[:1], r_data[:1], x_data[:1])
    assert pred_inj.shape == (1, n_buses, n_phases)

    packed_inj = predictor.pack_imputed_injections(pred_inj)
    assert isinstance(packed_inj, Injection)
    assert len(packed_inj.power_real.ids) == n_buses * n_phases
    assert len(packed_inj.power_real.values) == n_buses * n_phases


def test_predictor_save_and_load(mock_topology: Topology, tmp_path):
    """Test model persistence to disk."""
    predictor = PowerSystemInjectionPredictor(n_buses=132, n_phases=3)
    predictor.init_from_topology(mock_topology)

    v_data = np.random.normal(2400.0, 10.0, (2, 132, 3))
    l_data = np.random.normal(10.0, 2.0, (2, 132, 3))
    r_data = np.full((2, 131, 3, 3), 0.05)
    x_data = np.full((2, 131, 3, 3), 0.15)
    inj_data = np.random.normal(50.0, 5.0, (2, 132, 3))

    predictor.train(v_data, l_data, r_data, x_data, inj_data, epochs=1, batch_size=2)

    save_file = tmp_path / "model.h5"
    predictor.save_model(save_file)
    assert save_file.exists()
    assert (tmp_path / "params.pkl").exists()

    loaded_predictor = PowerSystemInjectionPredictor(n_buses=132, n_phases=3)
    loaded_predictor.load_model(save_file)

    pred = loaded_predictor.predict(v_data[:1], l_data[:1], r_data[:1], x_data[:1])
    assert pred.shape == (1, 132, 3)
