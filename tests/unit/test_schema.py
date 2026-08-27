"""Unit tests for schema.json and component_definition.json synchronization."""

import json
from pathlib import Path

from imputation_federate.schemas import ComponentParameters, DynamicInputs, DynamicOutputs, StaticInputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_schema_and_component_definition_sync() -> None:
    """Update root folder schema.json and verify synchronization with component_definition.json."""
    schema_path = PROJECT_ROOT / "schema.json"
    comp_def_path = PROJECT_ROOT / "component_definition.json"

    assert comp_def_path.exists(), f"component_definition.json not found at {comp_def_path}"

    # 1. Update/generate root folder schema.json
    StaticInputs.generate_json_schema(schema_path)

    # 2. Verify schema.json matches StaticInputs and ComponentParameters JSON schema
    model_schema = StaticInputs.model_json_schema()
    with open(schema_path, encoding="utf-8") as f:
        on_disk_schema = json.load(f)

    assert on_disk_schema == model_schema
    assert on_disk_schema == ComponentParameters.model_json_schema()

    # 3. Load component_definition.json and verify static_inputs match schema properties
    with open(comp_def_path, encoding="utf-8") as f:
        comp_def = json.load(f)

    static_inputs = comp_def.get("static_inputs", [])
    static_input_names = {item["port_id"] for item in static_inputs}
    schema_properties = set(model_schema.get("properties", {}).keys())

    missing = schema_properties - static_input_names
    extra = static_input_names - schema_properties
    assert static_input_names == schema_properties, (
        "Mismatch between component_definition.json static_inputs and StaticInputs schema properties.\n"
        f"Missing in component_definition.json: {missing}\n"
        f"Extra in component_definition.json: {extra}"
    )

    # 4. Verify component_definition.json dynamic_inputs match DynamicInputs fields
    dynamic_inputs = comp_def.get("dynamic_inputs", [])
    comp_def_dyn_inputs = {item["port_id"] for item in dynamic_inputs}
    model_dyn_inputs = set(DynamicInputs.model_fields.keys())
    assert comp_def_dyn_inputs == model_dyn_inputs, (
        f"Mismatch in dynamic_inputs: {comp_def_dyn_inputs ^ model_dyn_inputs}"
    )

    # 5. Verify dynamic outputs match DynamicOutputs fields
    dynamic_outputs = comp_def.get("dynamic_outputs", [])
    comp_def_dyn_outputs = {item["port_id"] for item in dynamic_outputs}
    model_dyn_outputs = set(DynamicOutputs.model_fields.keys())
    assert comp_def_dyn_outputs == model_dyn_outputs, (
        f"Mismatch in dynamic_outputs: {comp_def_dyn_outputs ^ model_dyn_outputs}"
    )
