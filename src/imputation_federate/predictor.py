"""Neural network-based power system injection and measurement predictor."""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from oedisi.types.data_types import Injection, PowersImaginary, PowersReal, Topology, VoltagesMagnitude

logger = logging.getLogger(__name__)

# Lazy TensorFlow / Keras import to speed up startup where possible
_tf_keras = None


def get_keras():
    """Lazily import keras."""
    global _tf_keras
    if _tf_keras is None:
        try:
            import keras

            _tf_keras = keras
        except ImportError:
            import tensorflow.keras as keras  # type: ignore

            _tf_keras = keras
    return _tf_keras


class PowerSystemInjectionPredictor:
    """FFNN Predictor for power system injection imputation."""

    def __init__(self, n_buses: int = 132, n_phases: int = 3, model_dir: str | Path | None = None) -> None:
        self.n_buses = n_buses
        self.n_phases = n_phases
        self.model_dir = Path(model_dir) if model_dir else None
        self.model: Any = None
        self.nonZeroIdx: np.ndarray | None = None
        self.totInjCol: int = n_buses * n_phases
        self.bus_names: list[str] = []
        self.branch_names: list[str] = []
        self.base_r: np.ndarray | None = None
        self.base_x: np.ndarray | None = None

    def _detectAbsentNodes(self, array_2d: np.ndarray) -> list[int]:
        """Detect columns that are all zeros or absent in the reference tensor."""
        if array_2d.shape[0] == 0:
            return []
        first_row = array_2d[0]
        return [i for i, value in enumerate(first_row) if value == 0 or np.isnan(value)]

    def _removeAbsentNodes(self, array: np.ndarray, columns_to_remove: list[int]) -> np.ndarray:
        """Remove absent columns from 2D array."""
        if not columns_to_remove or array.shape[1] <= max(columns_to_remove):
            return array
        return np.delete(array, columns_to_remove, axis=1)

    def _modifyAbsentPhases(self, array_2d: np.ndarray, base_array: np.ndarray | None = None) -> np.ndarray:
        """Filter out missing phase columns based on base reference tensor."""
        if base_array is None:
            base_array = array_2d
        cols_to_remove = self._detectAbsentNodes(base_array)
        return self._removeAbsentNodes(array_2d, cols_to_remove)

    def prepare_features(
        self,
        voltage_data: np.ndarray,
        load_forecast_data: np.ndarray,
        r_data: np.ndarray,
        x_data: np.ndarray,
        injection_data: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Prepare and flatten input feature tensor for neural network ingestion."""
        n_samples = voltage_data.shape[0]
        voltage_flat = voltage_data.reshape(n_samples, self.n_buses * self.n_phases)
        load_flat = load_forecast_data.reshape(n_samples, self.n_buses * self.n_phases)

        n_branches = r_data.shape[1]
        r_flat = r_data.reshape(n_samples, n_branches * self.n_phases * self.n_phases)
        x_flat = x_data.reshape(n_samples, n_branches * self.n_phases * self.n_phases)

        voltage_processed = self._modifyAbsentPhases(voltage_flat)
        load_processed = self._modifyAbsentPhases(load_flat, base_array=voltage_flat)
        r_processed = self._modifyAbsentPhases(r_flat)
        x_processed = self._modifyAbsentPhases(x_flat)

        x_feat = np.hstack([voltage_processed, load_processed, r_processed, x_processed])

        y_target = None
        if injection_data is not None:
            inj_flat = injection_data.reshape(n_samples, self.n_buses * self.n_phases)
            if self.nonZeroIdx is None:
                # Find columns that have any non-zero value across samples
                active_cols = np.where(np.any(inj_flat != 0, axis=0))[0]
                self.nonZeroIdx = active_cols if len(active_cols) > 0 else np.arange(inj_flat.shape[1])
                self.totInjCol = inj_flat.shape[1]

            mask = np.zeros(inj_flat.shape[1], dtype=bool)
            mask[self.nonZeroIdx] = True
            y_target = inj_flat[:, mask]

        return x_feat, y_target

    def build_model(self, input_dim: int, output_dim: int) -> Any:
        """Build the FFNN architecture."""
        keras = get_keras()
        inputs = keras.Input(shape=(input_dim,))

        x = keras.layers.BatchNormalization()(inputs)
        x = keras.layers.Dense(2048, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.15)(x)

        x = keras.layers.Dense(1500, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.12)(x)

        x = keras.layers.Dense(1024, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)

        x = keras.layers.Dense(768, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.08)(x)

        x = keras.layers.Dense(512, activation="tanh")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)

        x = keras.layers.Dense(100, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)

        outputs = keras.layers.Dense(output_dim, activation="relu")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error", metrics=["mae"])
        self.model = model
        return model

    def train(
        self,
        voltage_data: np.ndarray,
        load_forecast_data: np.ndarray,
        r_data: np.ndarray,
        x_data: np.ndarray,
        injection_data: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> Any:
        """Train the model with input tensors."""
        x_feat, y_target = self.prepare_features(
            voltage_data, load_forecast_data, r_data, x_data, injection_data=injection_data
        )
        assert y_target is not None

        if self.model is None:
            self.build_model(x_feat.shape[1], y_target.shape[1])

        history = self.model.fit(
            x_feat,
            y_target,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )
        return history

    def predict(
        self,
        voltage_data: np.ndarray,
        load_forecast_data: np.ndarray,
        r_data: np.ndarray,
        x_data: np.ndarray,
    ) -> np.ndarray:
        """Predict net injections tensor with shape (n_samples, n_buses, n_phases)."""
        if self.model is None:
            raise RuntimeError("Model is not initialized or loaded. Call load_model() first.")

        x_feat, _ = self.prepare_features(voltage_data, load_forecast_data, r_data, x_data)
        y_pred_filtered = self.model.predict(x_feat, verbose=0)

        n_samples = y_pred_filtered.shape[0]
        y_pred_full = np.zeros((n_samples, self.totInjCol))
        if self.nonZeroIdx is not None and len(self.nonZeroIdx) == y_pred_filtered.shape[1]:
            y_pred_full[:, self.nonZeroIdx] = y_pred_filtered
        else:
            limit = min(self.totInjCol, y_pred_filtered.shape[1])
            y_pred_full[:, :limit] = y_pred_filtered[:, :limit]

        return y_pred_full.reshape(-1, self.n_buses, self.n_phases)

    def save_model(self, model_path: str | Path) -> None:
        """Save model weights and metadata parameters."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save(str(path))

        params_file = path.parent / "params.pkl"
        params_data = {
            "nonZeroIdx": self.nonZeroIdx,
            "totInjCol": self.totInjCol,
            "n_buses": self.n_buses,
            "n_phases": self.n_phases,
            "bus_names": self.bus_names,
            "branch_names": self.branch_names,
        }
        with open(params_file, "wb") as f:
            pickle.dump(params_data, f)
        logger.info(f"Saved model to {path} and parameters to {params_file}")

    def load_model(self, model_path: str | Path) -> None:
        """Load trained neural network model and metadata parameters."""
        path = Path(model_path)
        keras = get_keras()
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = keras.models.load_model(str(path))

        params_file = path.parent / "params.pkl"
        if params_file.exists():
            with open(params_file, "rb") as f:
                params_data = pickle.load(f)
            self.nonZeroIdx = params_data.get("nonZeroIdx")
            self.totInjCol = params_data.get("totInjCol", self.n_buses * self.n_phases)
            self.n_buses = params_data.get("n_buses", self.n_buses)
            self.n_phases = params_data.get("n_phases", self.n_phases)
            self.bus_names = params_data.get("bus_names", [])
            self.branch_names = params_data.get("branch_names", [])
        logger.info(f"Successfully loaded model from {path}")

    # ── OEDISI Adapters ──────────────────────────────────────────────────────────

    def init_from_topology(self, topology: Topology) -> None:
        """Initialize internal grid structure, bus order, and branch impedance matrices from Topology."""
        buses = set()
        for bus_id in topology.base_voltage_magnitudes.ids:
            bus_name = bus_id.split(".")[0]
            buses.add(bus_name)
        self.bus_names = sorted(list(buses))
        self.n_buses = len(self.bus_names) if len(self.bus_names) > 0 else self.n_buses

        # Extract branches from admittance list
        branches = set()
        admittance = topology.admittance
        for fr, to in zip(admittance.from_equipment, admittance.to_equipment, strict=False):
            fr_b = fr.split(".")[0]
            to_b = to.split(".")[0]
            if fr_b != to_b:
                branches.add(f"{fr_b}_{to_b}")

        self.branch_names = sorted(list(branches))
        n_branches = max(1, len(self.branch_names))

        # Default symmetric base R and X
        self.base_r = np.full((n_branches, self.n_phases, self.n_phases), 0.05)
        self.base_x = np.full((n_branches, self.n_phases, self.n_phases), 0.15)
        for b in range(n_branches):
            self.base_r[b] = (self.base_r[b] + self.base_r[b].T) / 2.0
            self.base_x[b] = (self.base_x[b] + self.base_x[b].T) / 2.0

        logger.info(
            f"Initialized grid structure from topology: {self.n_buses} buses, {len(self.branch_names)} branches"
        )

    def extract_voltage_tensor(self, voltages_mag: VoltagesMagnitude) -> np.ndarray:
        """Map VoltagesMagnitude into (1, n_buses, n_phases) tensor."""
        v_tensor = np.zeros((1, self.n_buses, self.n_phases))
        bus_map = {name: i for i, name in enumerate(self.bus_names)}

        for node_id, val in zip(voltages_mag.ids, voltages_mag.values, strict=False):
            parts = node_id.split(".")
            bus_name = parts[0]
            phase_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1].isdigit() else 0
            if bus_name in bus_map and 0 <= phase_idx < self.n_phases:
                v_tensor[0, bus_map[bus_name], phase_idx] = val

        return v_tensor

    def extract_power_tensor(
        self,
        powers_real: PowersReal | None,
        injections: Injection | None = None,
    ) -> np.ndarray:
        """Map PowersReal or Injection into (1, n_buses, n_phases) tensor."""
        p_tensor = np.zeros((1, self.n_buses, self.n_phases))
        bus_map = {name: i for i, name in enumerate(self.bus_names)}

        real_data = powers_real
        if real_data is None and injections is not None and injections.power_real is not None:
            real_data = injections.power_real

        if real_data is not None:
            for node_id, val in zip(real_data.ids, real_data.values, strict=False):
                parts = node_id.split(".")
                bus_name = parts[0]
                phase_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1].isdigit() else 0
                if bus_name in bus_map and 0 <= phase_idx < self.n_phases:
                    p_tensor[0, bus_map[bus_name], phase_idx] = val

        return p_tensor

    def pack_imputed_injections(
        self,
        predicted_inj: np.ndarray,
        time_val: Any = None,
    ) -> Injection:
        """Pack predicted injection tensor (1, n_buses, n_phases) into OEDISI Injection model."""
        ids = []
        equipment_ids = []
        real_values = []
        imag_values = []

        arr = predicted_inj[0]
        for b_idx, bus_name in enumerate(self.bus_names):
            for p_idx in range(self.n_phases):
                phase_num = p_idx + 1
                node_id = f"{bus_name}.{phase_num}"
                p_val = float(arr[b_idx, p_idx])
                ids.append(node_id)
                equipment_ids.append(f"Load.{bus_name}_{phase_num}")
                real_values.append(p_val)
                # Nominal power factor assumption (0.95 lag => Q ~ 0.33 P)
                imag_values.append(p_val * 0.32868)

        power_real = PowersReal(
            ids=ids,
            equipment_ids=equipment_ids,
            values=real_values,
            units="kW",
            time=time_val,
        )
        power_imag = PowersImaginary(
            ids=ids,
            equipment_ids=equipment_ids,
            values=imag_values,
            units="kVAR",
            time=time_val,
        )

        return Injection(power_real=power_real, power_imaginary=power_imag)

    def pack_imputed_voltages(
        self,
        measured_voltages: VoltagesMagnitude,
        predicted_inj: np.ndarray | None = None,
    ) -> VoltagesMagnitude:
        """Reconstruct full VoltagesMagnitude with imputed missing nodes."""
        measured_map = {
            node_id: val for node_id, val in zip(measured_voltages.ids, measured_voltages.values, strict=False)
        }

        full_ids = []
        full_values = []

        for bus_name in self.bus_names:
            for p_idx in range(self.n_phases):
                node_id = f"{bus_name}.{p_idx + 1}"
                if node_id in measured_map and measured_map[node_id] > 0:
                    val = measured_map[node_id]
                else:
                    # Impute missing voltage as nominal 2401.77 V (~1.0 pu on 4.16 kV LL)
                    val = 2401.77
                full_ids.append(node_id)
                full_values.append(val)

        return VoltagesMagnitude(ids=full_ids, values=full_values, units="V", time=measured_voltages.time)
