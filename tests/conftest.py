"""Shared fixtures and mock data for imputation federate tests."""

import os

# Disable CUDA device discovery and quiet TensorFlow to speed up testing
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Initialize deep learning runtime before CFFI / HELICS libraries
try:
    import keras  # noqa: F401
except ImportError:
    pass

import pytest
from oedisi.types.data_types import (
    AdmittanceSparse,
    Injection,
    PowersImaginary,
    PowersReal,
    Topology,
    VoltagesAngle,
    VoltagesMagnitude,
)


@pytest.fixture
def mock_topology() -> Topology:
    """Mock IEEE 123 Topology fixture."""
    node_ids = [f"{i}.{p}" for i in range(1, 133) for p in [1, 2, 3]]
    base_v = [2401.77] * len(node_ids)
    base_ang = [0.0 if ".1" in n else (-2.09 if ".2" in n else 2.09) for n in node_ids]

    admittance = AdmittanceSparse(
        from_equipment=[f"{i}.1" for i in range(1, 132)],
        to_equipment=[f"{i + 1}.1" for i in range(1, 132)],
        admittance_list=[(10.0, -20.0)] * 131,
    )

    power_real = PowersReal(
        ids=node_ids,
        equipment_ids=[f"Load.{n}" for n in node_ids],
        values=[5.0] * len(node_ids),
        units="kW",
    )
    power_imag = PowersImaginary(
        ids=node_ids,
        equipment_ids=[f"Load.{n}" for n in node_ids],
        values=[1.5] * len(node_ids),
        units="kVAR",
    )
    inj = Injection(power_real=power_real, power_imaginary=power_imag)

    return Topology(
        admittance=admittance,
        base_voltage_magnitudes=VoltagesMagnitude(ids=node_ids, values=base_v, units="V"),
        base_voltage_angles=VoltagesAngle(ids=node_ids, values=base_ang),
        injections=inj,
        slack_bus=["150.1", "150.2", "150.3"],
        incidences=None,
    )


@pytest.fixture
def mock_voltages_magnitude() -> VoltagesMagnitude:
    """Mock VoltagesMagnitude fixture with 30% missing nodes to test imputation."""
    all_nodes = [f"{i}.{p}" for i in range(1, 133) for p in [1, 2, 3]]
    # Keep subset (70%)
    measured_nodes = all_nodes[:280]
    values = [2400.0 + (i % 20) for i in range(len(measured_nodes))]
    return VoltagesMagnitude(ids=measured_nodes, values=values, units="V")


@pytest.fixture
def mock_injections() -> Injection:
    """Mock Injection fixture with missing/corrupted nodes."""
    all_nodes = [f"{i}.{p}" for i in range(1, 133) for p in [1, 2, 3]]
    measured_nodes = all_nodes[:200]
    power_real = PowersReal(
        ids=measured_nodes,
        equipment_ids=[f"Load.{n}" for n in measured_nodes],
        values=[12.5] * len(measured_nodes),
        units="kW",
    )
    power_imag = PowersImaginary(
        ids=measured_nodes,
        equipment_ids=[f"Load.{n}" for n in measured_nodes],
        values=[4.0] * len(measured_nodes),
        units="kVAR",
    )
    return Injection(power_real=power_real, power_imaginary=power_imag)
