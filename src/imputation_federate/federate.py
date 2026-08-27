import logging
from pathlib import Path
from typing import Any

import helics as h
import numpy as np
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import Injection, PowersImaginary, PowersReal, Topology, VoltagesMagnitude

from .predictor import PowerSystemInjectionPredictor
from .schemas import ComponentDefinition, StaticInputs

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


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
        fedinfo = h.helicsCreateFederateInfo()
        fedinfo.core_name = self.static.name
        fedinfo.core_type = core_type
        fedinfo.core_init = "--federates=1"

        if broker_name is not None:
            h.helicsFederateInfoSetBroker(fedinfo, broker_name)
        elif broker_config is not None:
            h.helicsFederateInfoSetBroker(fedinfo, broker_config.broker_ip)
            h.helicsFederateInfoSetBrokerPort(fedinfo, broker_config.broker_port)

        h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_delta, self.static.deltat)

        self.vfed = h.helicsCreateValueFederate(self.static.name, fedinfo)
        logger.info(f"Created HELICS value federate '{self.static.name}'")

    def _register_interfaces(self) -> None:
        """Register subscriptions and publications."""
        dyn_in = self.config.dynamic_inputs
        dyn_out = self.config.dynamic_outputs

        # Subscriptions
        if dyn_in.topology:
            self.sub.topology = h.helicsFederateRegisterSubscription(self.vfed, dyn_in.topology, "")
            h.helicsInputSetOption(self.sub.topology, h.HELICS_HANDLE_OPTION_CONNECTION_OPTIONAL, 1)

        if dyn_in.voltages_magnitude:
            self.sub.voltages_magnitude = h.helicsFederateRegisterSubscription(self.vfed, dyn_in.voltages_magnitude, "")
            h.helicsInputSetOption(self.sub.voltages_magnitude, h.HELICS_HANDLE_OPTION_CONNECTION_OPTIONAL, 1)

        if dyn_in.injections:
            self.sub.injections = h.helicsFederateRegisterSubscription(self.vfed, dyn_in.injections, "")
            h.helicsInputSetOption(self.sub.injections, h.HELICS_HANDLE_OPTION_CONNECTION_OPTIONAL, 1)

        if dyn_in.powers_real:
            self.sub.powers_real = h.helicsFederateRegisterSubscription(self.vfed, dyn_in.powers_real, "")
            h.helicsInputSetOption(self.sub.powers_real, h.HELICS_HANDLE_OPTION_CONNECTION_OPTIONAL, 1)

        if dyn_in.powers_imag:
            self.sub.powers_imag = h.helicsFederateRegisterSubscription(self.vfed, dyn_in.powers_imag, "")
            h.helicsInputSetOption(self.sub.powers_imag, h.HELICS_HANDLE_OPTION_CONNECTION_OPTIONAL, 1)

        # Publications
        if dyn_out.injections:
            self.pub.injections = h.helicsFederateRegisterPublication(
                self.vfed, dyn_out.injections, h.HELICS_DATA_TYPE_STRING, ""
            )

        if dyn_out.voltages_magnitude:
            self.pub.voltages_magnitude = h.helicsFederateRegisterPublication(
                self.vfed, dyn_out.voltages_magnitude, h.HELICS_DATA_TYPE_STRING, ""
            )

        if dyn_out.powers_real:
            self.pub.powers_real = h.helicsFederateRegisterPublication(
                self.vfed, dyn_out.powers_real, h.HELICS_DATA_TYPE_STRING, ""
            )

        if dyn_out.powers_imag:
            self.pub.powers_imag = h.helicsFederateRegisterPublication(
                self.vfed, dyn_out.powers_imag, h.HELICS_DATA_TYPE_STRING, ""
            )

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
                if self.sub.topology is not None and h.helicsInputIsUpdated(self.sub.topology):
                    raw_topo = h.helicsInputGetString(self.sub.topology)
                    if raw_topo:
                        try:
                            topo_data = Topology.model_validate_json(raw_topo)
                            self.predictor.init_from_topology(topo_data)
                        except Exception as e:
                            logger.error(f"Error parsing topology: {e}")

                # 2. Update Measurement Arrays
                if self.sub.voltages_magnitude is not None and h.helicsInputIsUpdated(self.sub.voltages_magnitude):
                    raw_v = h.helicsInputGetString(self.sub.voltages_magnitude)
                    if raw_v:
                        try:
                            latest_voltages = VoltagesMagnitude.model_validate_json(raw_v)
                        except Exception as e:
                            logger.error(f"Error parsing voltages_magnitude: {e}")

                if self.sub.injections is not None and h.helicsInputIsUpdated(self.sub.injections):
                    raw_inj = h.helicsInputGetString(self.sub.injections)
                    if raw_inj:
                        try:
                            latest_injections = Injection.model_validate_json(raw_inj)
                        except Exception as e:
                            logger.error(f"Error parsing injections: {e}")

                if self.sub.powers_real is not None and h.helicsInputIsUpdated(self.sub.powers_real):
                    raw_pr = h.helicsInputGetString(self.sub.powers_real)
                    if raw_pr:
                        try:
                            latest_powers_real = PowersReal.model_validate_json(raw_pr)
                        except Exception as e:
                            logger.error(f"Error parsing powers_real: {e}")

                if self.sub.powers_imag is not None and h.helicsInputIsUpdated(self.sub.powers_imag):
                    raw_pi = h.helicsInputGetString(self.sub.powers_imag)
                    if raw_pi:
                        try:
                            latest_powers_imag = PowersImaginary.model_validate_json(raw_pi)
                        except Exception as e:
                            logger.error(f"Error parsing powers_imag: {e}")

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
                    if self.pub.injections is not None:
                        h.helicsPublicationPublishString(self.pub.injections, imputed_injections.model_dump_json())

                    if self.pub.voltages_magnitude is not None:
                        h.helicsPublicationPublishString(
                            self.pub.voltages_magnitude, imputed_voltages.model_dump_json()
                        )

                    if self.pub.powers_real is not None and imputed_injections.power_real is not None:
                        h.helicsPublicationPublishString(
                            self.pub.powers_real, imputed_injections.power_real.model_dump_json()
                        )

                    if self.pub.powers_imag is not None and imputed_injections.power_imaginary is not None:
                        # If latest_powers_imag was supplied, preserve its scaling or use imputed reactive power
                        if latest_powers_imag is not None and len(latest_powers_imag.values) == len(
                            imputed_injections.power_imaginary.values
                        ):
                            pub_imag = latest_powers_imag
                        else:
                            pub_imag = imputed_injections.power_imaginary
                        h.helicsPublicationPublishString(self.pub.powers_imag, pub_imag.model_dump_json())

                granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

        except Exception as e:
            logger.exception(f"Simulator background task failed with unhandled exception: {e}")
            raise
        finally:
            self.destroy()

    def destroy(self) -> None:
        """Clean up HELICS resources and disconnect federate."""
        if self.vfed is not None:
            logger.info("Disconnecting and freeing HELICS federate allocations...")
            try:
                h.helicsFederateDisconnect(self.vfed)
                h.helicsFederateFree(self.vfed)
            except Exception as e:
                logger.warning(f"Error during federate disconnect: {e}")
            self.vfed = None
        h.helicsCloseLibrary()
        logger.info("HELICS library closed.")


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
