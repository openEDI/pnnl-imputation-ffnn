"""Pydantic configuration models and validation schemas for the Imputation Federate."""

import json
from pathlib import Path
from typing import Any

from oedisi.types.common import DefaultFileNames
from oedisi.types.helics_config import HELICSFederateConfig
from pydantic import BaseModel, Field


class StaticInputs(HELICSFederateConfig):
    """Static configuration parameters for the Imputation Federate."""

    name: str = Field(..., description="Unique identifier for the federate instance")
    model_name: str = Field("ieee123", description="Model architecture/feeder key (e.g. 'ieee123')")
    model_path: str = Field("", description="Optional custom directory path to trained model and params")
    deltat: float = Field(0.1, ge=0.0, description="HELICS time step delta in seconds")
    end_time: float = Field(86400.0, ge=0.0, description="Maximum simulation end time in seconds")
    run_freq_sec: float = Field(900.0, ge=0.0, description="Execution frequency in seconds")

    model_config = {
        "title": "ImputationConfig",
        "populate_by_name": True,
    }


class DynamicInputs(BaseModel):
    """Dynamic input subscriptions required for federate execution."""

    topology: str = Field(..., description="Subscription topic for network topology")
    voltages_magnitude: str = Field(..., description="Subscription topic for measured voltage magnitudes")
    injections: str | None = Field(None, description="Optional subscription topic for measured injections")
    powers_real: str | None = Field(None, description="Optional subscription topic for measured real powers")
    powers_imag: str | None = Field(None, description="Optional subscription topic for measured imaginary powers")


class DynamicOutputs(BaseModel):
    """Dynamic output publications produced by the federate."""

    injections: str | None = Field(default="injections", description="Publication topic for imputed injections")
    voltages_magnitude: str | None = Field(
        default="voltages_magnitude", description="Publication topic for imputed voltage magnitudes"
    )
    powers_real: str | None = Field(default="powers_real", description="Publication topic for imputed real powers")
    powers_imag: str | None = Field(default="powers_imag", description="Publication topic for imputed imaginary powers")


class ComponentDefinition(BaseModel):
    """Unified component definition containing static inputs, dynamic inputs, and dynamic outputs."""

    static_inputs: StaticInputs
    dynamic_inputs: DynamicInputs
    dynamic_outputs: DynamicOutputs = Field(default_factory=DynamicOutputs)

    @classmethod
    def from_build_files(
        cls,
        static_inputs_path: str | Path = DefaultFileNames.STATIC_INPUTS.value,
        input_mapping_path: str | Path = DefaultFileNames.INPUT_MAPPING.value,
    ) -> "ComponentDefinition":
        """Construct and validate ComponentDefinition from static_inputs.json and input_mapping.json."""
        static_path = Path(static_inputs_path)
        mapping_path = Path(input_mapping_path)

        if not static_path.exists():
            raise FileNotFoundError(f"Static inputs file not found: {static_path}")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Input mapping file not found: {mapping_path}")

        with open(static_path, encoding="utf-8") as fh:
            raw_static = json.load(fh)

        with open(mapping_path, encoding="utf-8") as fh:
            raw_mapping: dict[str, Any] = json.load(fh)

        static_inputs = StaticInputs.model_validate(raw_static)
        dynamic_inputs = DynamicInputs.model_validate(raw_mapping)
        dynamic_outputs = DynamicOutputs.model_validate(raw_mapping)

        return cls(
            static_inputs=static_inputs,
            dynamic_inputs=dynamic_inputs,
            dynamic_outputs=dynamic_outputs,
        )
