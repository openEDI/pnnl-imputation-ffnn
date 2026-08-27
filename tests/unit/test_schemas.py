"""Unit tests for Pydantic configuration schemas."""

import json

import pytest
from pydantic import ValidationError

from imputation_federate.schemas import ComponentDefinition, DynamicInputs, StaticInputs


def test_static_inputs_defaults():
    """Verify default values and valid initialization of StaticInputs."""
    cfg = StaticInputs(name="test_imputation")
    assert cfg.name == "test_imputation"
    assert cfg.model_name == "ieee123"
    assert cfg.deltat == 0.1
    assert cfg.end_time == 86400.0


def test_static_inputs_validation_error():
    """Verify validation error on missing required field."""
    with pytest.raises(ValidationError):
        StaticInputs()  # type: ignore


def test_dynamic_inputs_required():
    """Verify that topology and voltages_magnitude are required."""
    with pytest.raises(ValidationError):
        DynamicInputs()  # type: ignore

    dyn = DynamicInputs(topology="feeder/topology", voltages_magnitude="sensor/voltages")
    assert dyn.topology == "feeder/topology"
    assert dyn.voltages_magnitude == "sensor/voltages"
    assert dyn.injections is None


def test_component_definition_from_build_files(tmp_path):
    """Verify from_build_files loads and validates properly."""
    static_file = tmp_path / "static_inputs.json"
    mapping_file = tmp_path / "input_mapping.json"

    static_data = {"name": "imp_fed", "model_name": "ieee123", "deltat": 0.5}
    mapping_data = {
        "topology": "feeder/topology",
        "voltages_magnitude": "sensor_v/publication",
        "injections": "sensor_inj/publication",
    }

    static_file.write_text(json.dumps(static_data))
    mapping_file.write_text(json.dumps(mapping_data))

    comp_def = ComponentDefinition.from_build_files(
        static_inputs_path=static_file,
        input_mapping_path=mapping_file,
    )
    assert comp_def.static_inputs.name == "imp_fed"
    assert comp_def.static_inputs.deltat == 0.5
    assert comp_def.dynamic_inputs.topology == "feeder/topology"
    assert comp_def.dynamic_inputs.voltages_magnitude == "sensor_v/publication"
    assert comp_def.dynamic_inputs.injections == "sensor_inj/publication"
