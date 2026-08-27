import logging
from pathlib import Path
from typing import Any, TypeVar

import helics as h
import numpy as np
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import Injection, PowersImaginary, PowersReal, Topology, VoltagesMagnitude
from oedisi.types.helics_config import HELICSBrokerConfig
from pydantic import BaseModel

from .predictor import PowerSystemInjectionPredictor
from .schemas import ComponentDefinition, StaticInputs

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# HELICS & OEDISI Interface Helpers
# =============================================================================


def create_federate_info(
    static_cfg: StaticInputs,
    broker_config: BrokerConfig | None = None,
    core_type: int = h.HELICS_CORE_TYPE_ZMQ,
    broker_name: str | None = None,
) -> Any:
    """Construct and configure a HELICS FederateInfo object from StaticInputs and BrokerConfig."""
    fedinfo = h.helicsCreateFederateInfo()
    fedinfo.core_name = static_cfg.name
    fedinfo.core_type = core_type
    fedinfo.core_init = "--federates=1"

    # Apply broker parameters via HELICSBrokerConfig if provided
    if broker_name is not None:
        h.helicsFederateInfoSetBroker(fedinfo, broker_name)
    elif broker_config is not None:
        broker_meta = HELICSBrokerConfig.from_rest_config(broker_config)
        if broker_meta.host is not None:
            h.helicsFederateInfoSetBroker(fedinfo, broker_meta.host)
        if broker_meta.port is not None:
            h.helicsFederateInfoSetBrokerPort(fedinfo, broker_meta.port)

    # Apply configuration defaults (core_type, core_init, broker settings) if supported
    if hasattr(static_cfg, "apply_to_federate_info"):
        static_cfg.apply_to_federate_info(fedinfo)

    # Set simulation time delta property
    h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_delta, static_cfg.deltat)

    return fedinfo


def register_subscription(vfed: Any, key: str, optional: bool = True) -> Any:
    """Register a HELICS subscription handle with optional connection flags."""
    sub = h.helicsFederateRegisterSubscription(vfed, key, "")
    if optional:
        h.helicsInputSetOption(sub, h.HELICS_HANDLE_OPTION_CONNECTION_OPTIONAL, 1)
    return sub


def register_publication(vfed: Any, key: str, data_type: int = h.HELICS_DATA_TYPE_STRING) -> Any:
    """Register a HELICS publication handle."""
    return h.helicsFederateRegisterPublication(vfed, key, data_type, "")


def get_subscription_data(input_handle: Any, model_cls: type[T]) -> T | None:
    """Read, parse, and validate updated data from a HELICS subscription handle using Pydantic."""
    if input_handle is not None and h.helicsInputIsUpdated(input_handle):
        raw = h.helicsInputGetString(input_handle)
        if raw:
            try:
                return model_cls.model_validate_json(raw)
            except Exception as e:
                logger.error(f"Error parsing {model_cls.__name__} from subscription: {e}")
    return None


def publish_data(pub_handle: Any, data: BaseModel | None) -> None:
    """Serialize a Pydantic model to JSON and publish to a HELICS publication handle."""
    if pub_handle is not None and data is not None:
        h.helicsPublicationPublishString(pub_handle, data.model_dump_json())


def cleanup_federate(vfed: Any) -> None:
    """Safely disconnect and free a HELICS federate handle and close the HELICS library."""
    if vfed is not None:
        logger.info("Disconnecting and freeing HELICS federate allocations...")
        try:
            h.helicsFederateDisconnect(vfed)
            h.helicsFederateFree(vfed)
        except Exception as e:
            logger.warning(f"Error during federate disconnect: {e}")
    h.helicsCloseLibrary()
    logger.info("HELICS library closed.")


# =============================================================================
# Federate Class & Lifecycle
# =============================================================================


class Subscriptions:
    """Holder for the federate's HELICS input handles."""

    topology: Any = None
    voltages_magnitude: Any = None
    injections: Any = None
    powers_real: Any = None
    powers_imag: Any = None


class Publications:
    """Holder for the federate's HELICS output handles."""

    injections: Any = None
    voltages_magnitude: Any = None
    powers_real: Any = None
    powers_imag: Any = None


class ImputationFederate:
    """OEDISI Value Federate for FFNN-based Imputation."""

    def __init__(
        self,
        config: ComponentDefinition,
        broker_config: BrokerConfig | None = None,
        core_type: int = h.HELICS_CORE_TYPE_ZMQ,
        broker_name: str | None = None,
    ) -> None:
        self.config = config
        self.static: StaticInputs = config.static_inputs
        self.vfed: Any = None
        self.sub = Subscriptions()
        self.pub = Publications()
        self.predictor = PowerSystemInjectionPredictor(n_buses=132, n_phases=3)

        # Locate model weights
        self._init_predictor()
        self._init_federate(broker_config, core_type=core_type, broker_name=broker_name)
        self._register_interfaces()

    def _init_predictor(self) -> None:
        """Find and load pre-trained model checkpoint or initialize default."""
        model_path = None
        if self.static.model_path:
            p = Path(self.static.model_path)
            if p.exists():
                model_path = p

        if model_path is None:
            # Check default package models directory
            pkg_model = (
                Path(__file__).parent / "models" / self.static.model_name / "power_system_injection_predictor.h5"
            )
            if pkg_model.exists():
                model_path = pkg_model

        if model_path and model_path.exists():
            try:
                self.predictor.load_model(model_path)
                logger.info(f"Loaded imputation model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load pre-trained model from {model_path}: {e}")
        else:
            logger.info("No pre-trained model file found. Building synthetic baseline architecture.")
            # Build and initialize baseline model
            self.predictor.build_model(input_dim=1584, output_dim=132)

    def _init_federate(
        self,
        broker_config: BrokerConfig | None,
        core_type: int = h.HELICS_CORE_TYPE_ZMQ,
        broker_name: str | None = None,
    ) -> None:
        """Create and configure HELICS Value Federate."""
        fedinfo = create_federate_info(
            self.static, broker_config=broker_config, core_type=core_type, broker_name=broker_name
        )
        self.vfed = h.helicsCreateValueFederate(self.static.name, fedinfo)
        logger.info(f"Created HELICS value federate '{self.static.name}'")

    def _register_interfaces(self) -> None:
        """Register subscriptions and publications using helper functions."""
        dyn_in = self.config.dynamic_inputs
        dyn_out = self.config.dynamic_outputs

        # Subscriptions
        if dyn_in.topology:
            self.sub.topology = register_subscription(self.vfed, dyn_in.topology)

        if dyn_in.voltages_magnitude:
            self.sub.voltages_magnitude = register_subscription(self.vfed, dyn_in.voltages_magnitude)

        if dyn_in.injections:
            self.sub.injections = register_subscription(self.vfed, dyn_in.injections)

        if dyn_in.powers_real:
            self.sub.powers_real = register_subscription(self.vfed, dyn_in.powers_real)

        if dyn_in.powers_imag:
            self.sub.powers_imag = register_subscription(self.vfed, dyn_in.powers_imag)

        # Publications
        if dyn_out.injections:
            self.pub.injections = register_publication(self.vfed, dyn_out.injections)

        if dyn_out.voltages_magnitude:
            self.pub.voltages_magnitude = register_publication(self.vfed, dyn_out.voltages_magnitude)

        if dyn_out.powers_real:
            self.pub.powers_real = register_publication(self.vfed, dyn_out.powers_real)

        if dyn_out.powers_imag:
            self.pub.powers_imag = register_publication(self.vfed, dyn_out.powers_imag)

        logger.info("Successfully registered HELICS subscriptions and publications")

    def run(self) -> None:
        """Run the main HELICS simulation timing loop."""
        try:
            h.helicsFederateEnterExecutingMode(self.vfed)
            logger.info("Entered HELICS execution mode")

            granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

            latest_voltages: VoltagesMagnitude | None = None
            latest_powers_real: PowersReal | None = None
            latest_powers_imag: PowersImaginary | None = None
            latest_injections: Injection | None = None

            while granted_time < h.HELICS_TIME_MAXTIME:
                if granted_time > self.static.end_time:
                    logger.info(f"Granted time {granted_time} > end_time {self.static.end_time}. Terminating loop.")
                    break

                # 1. Update Topology (read-only structural reference)
                if (topo := get_subscription_data(self.sub.topology, Topology)) is not None:
                    self.predictor.init_from_topology(topo)

                # 2. Update Measurement Arrays
                if (v := get_subscription_data(self.sub.voltages_magnitude, VoltagesMagnitude)) is not None:
                    latest_voltages = v

                if (inj := get_subscription_data(self.sub.injections, Injection)) is not None:
                    latest_injections = inj

                if (pr := get_subscription_data(self.sub.powers_real, PowersReal)) is not None:
                    latest_powers_real = pr

                if (pi := get_subscription_data(self.sub.powers_imag, PowersImaginary)) is not None:
                    latest_powers_imag = pi

                # 3. Perform Imputation & Publish Outputs
                if latest_voltages is not None:
                    time_stamp = latest_voltages.time

                    # Prepare input tensors
                    v_tensor = self.predictor.extract_voltage_tensor(latest_voltages)
                    p_tensor = self.predictor.extract_power_tensor(latest_powers_real, latest_injections)

                    n_branches = max(1, len(self.predictor.branch_names))
                    r_tensor = np.tile(self.predictor.base_r or np.full((n_branches, 3, 3), 0.05), (1, 1, 1, 1))
                    x_tensor = np.tile(self.predictor.base_x or np.full((n_branches, 3, 3), 0.15), (1, 1, 1, 1))

                    try:
                        # Predict imputed net injections
                        predicted_inj = self.predictor.predict(v_tensor, p_tensor, r_tensor, x_tensor)
                    except Exception as err:
                        logger.warning(
                            f"Predictor inference failed, falling back to pass-through/reconstruction: {err}"
                        )
                        predicted_inj = p_tensor

                    # Reconstruct full clean measurement outputs
                    imputed_injections = self.predictor.pack_imputed_injections(predicted_inj, time_val=time_stamp)
                    imputed_voltages = self.predictor.pack_imputed_voltages(latest_voltages, predicted_inj)

                    # Publish to downstream algorithm federates
                    publish_data(self.pub.injections, imputed_injections)
                    publish_data(self.pub.voltages_magnitude, imputed_voltages)

                    if imputed_injections.power_real is not None:
                        publish_data(self.pub.powers_real, imputed_injections.power_real)

                    if imputed_injections.power_imaginary is not None:
                        # If latest_powers_imag was supplied, preserve its scaling or use imputed reactive power
                        if latest_powers_imag is not None and len(latest_powers_imag.values) == len(
                            imputed_injections.power_imaginary.values
                        ):
                            pub_imag = latest_powers_imag
                        else:
                            pub_imag = imputed_injections.power_imaginary
                        publish_data(self.pub.powers_imag, pub_imag)

                granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

        except Exception as e:
            logger.exception(f"Simulator background task failed with unhandled exception: {e}")
            raise
        finally:
            self.destroy()

    def destroy(self) -> None:
        """Clean up HELICS resources and disconnect federate."""
        if self.vfed is not None:
            cleanup_federate(self.vfed)
            self.vfed = None


def run_simulator(broker_config: BrokerConfig | None = None) -> None:
    """Entry point for background simulation task."""
    config = ComponentDefinition.from_build_files()
    federate = ImputationFederate(config, broker_config)
    federate.run()


def main() -> None:
    """CLI execution entrypoint."""
    run_simulator(BrokerConfig(broker_ip="127.0.0.1", broker_port=23404))


if __name__ == "__main__":
    main()
