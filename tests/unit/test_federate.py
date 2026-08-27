"""Unit tests for HELICS helper functions and ImputationFederate lifecycle."""

from unittest.mock import MagicMock, patch

import helics as h
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import PowersReal, VoltagesMagnitude

from imputation_federate.federate import (
    ImputationFederate,
    cleanup_federate,
    create_federate_info,
    get_subscription_data,
    publish_data,
    register_publication,
    register_subscription,
)
from imputation_federate.schemas import ComponentDefinition, DynamicInputs, DynamicOutputs, StaticInputs


def test_create_federate_info():
    """Verify that create_federate_info populates core, broker, and time properties correctly."""
    static = StaticInputs(
        name="test_fed",
        deltat=0.25,
        core_type="zmq",
    )
    broker_cfg = BrokerConfig(broker_ip="192.168.1.100", broker_port=23404)

    fedinfo = create_federate_info(static, broker_config=broker_cfg)
    assert fedinfo is not None


def test_register_subscription_and_publication():
    """Verify subscription and publication registration wrappers."""
    mock_fed = MagicMock()
    with (
        patch("helics.helicsFederateRegisterSubscription") as mock_reg_sub,
        patch("helics.helicsInputSetOption") as mock_set_opt,
        patch("helics.helicsFederateRegisterPublication") as mock_reg_pub,
    ):
        mock_reg_sub.return_value = "sub_handle_1"
        mock_reg_pub.return_value = "pub_handle_1"

        sub = register_subscription(mock_fed, "test/sub_topic", optional=True)
        assert sub == "sub_handle_1"
        mock_reg_sub.assert_called_once_with(mock_fed, "test/sub_topic", "")
        mock_set_opt.assert_called_once_with("sub_handle_1", h.HELICS_HANDLE_OPTION_CONNECTION_OPTIONAL, 1)

        pub = register_publication(mock_fed, "test/pub_topic")
        assert pub == "pub_handle_1"
        mock_reg_pub.assert_called_once_with(mock_fed, "test/pub_topic", h.HELICS_DATA_TYPE_STRING, "")


def test_get_subscription_data_updated():
    """Verify get_subscription_data deserializes valid Pydantic models when updated."""
    mock_input = MagicMock()
    voltages_sample = VoltagesMagnitude(
        values=[1.0, 0.99, 1.01],
        ids=["bus_1.1", "bus_1.2", "bus_1.3"],
        time=10.0,
    )
    json_str = voltages_sample.model_dump_json()

    with (
        patch("helics.helicsInputIsUpdated", return_value=True),
        patch("helics.helicsInputGetString", return_value=json_str),
    ):
        result = get_subscription_data(mock_input, VoltagesMagnitude)
        assert result is not None
        assert result.values == [1.0, 0.99, 1.01]
        assert result.ids == ["bus_1.1", "bus_1.2", "bus_1.3"]


def test_get_subscription_data_not_updated():
    """Verify get_subscription_data returns None when input is not updated."""
    mock_input = MagicMock()
    with patch("helics.helicsInputIsUpdated", return_value=False):
        result = get_subscription_data(mock_input, VoltagesMagnitude)
        assert result is None


def test_get_subscription_data_invalid_json():
    """Verify get_subscription_data handles malformed JSON without raising unhandled errors."""
    mock_input = MagicMock()
    with (
        patch("helics.helicsInputIsUpdated", return_value=True),
        patch("helics.helicsInputGetString", return_value="invalid-json-payload"),
    ):
        result = get_subscription_data(mock_input, VoltagesMagnitude)
        assert result is None


def test_publish_data():
    """Verify publish_data dumps Pydantic model to JSON and publishes string."""
    mock_pub = MagicMock()
    sample_power = PowersReal(values=[10.5, 20.2], ids=["inj_1", "inj_2"], equipment_ids=["eq_1", "eq_2"])

    with patch("helics.helicsPublicationPublishString") as mock_publish:
        publish_data(mock_pub, sample_power)
        mock_publish.assert_called_once_with(mock_pub, sample_power.model_dump_json())

    # None data should not call publish
    with patch("helics.helicsPublicationPublishString") as mock_publish:
        publish_data(mock_pub, None)
        mock_publish.assert_not_called()


def test_cleanup_federate():
    """Verify cleanup_federate calls disconnect, free, and close library safely."""
    mock_fed = MagicMock()
    with (
        patch("helics.helicsFederateDisconnect") as mock_disc,
        patch("helics.helicsFederateFree") as mock_free,
        patch("helics.helicsCloseLibrary") as mock_close,
    ):
        cleanup_federate(mock_fed)
        mock_disc.assert_called_once_with(mock_fed)
        mock_free.assert_called_once_with(mock_fed)
        mock_close.assert_called_once()


def test_cleanup_federate_exception_handling():
    """Verify cleanup_federate catches exceptions during disconnect and still closes library."""
    mock_fed = MagicMock()
    with (
        patch("helics.helicsFederateDisconnect", side_effect=RuntimeError("Helics error")),
        patch("helics.helicsFederateFree") as mock_free,
        patch("helics.helicsCloseLibrary") as mock_close,
    ):
        cleanup_federate(mock_fed)
        mock_free.assert_not_called()
        mock_close.assert_called_once()


def test_imputation_federate_init_and_destroy():
    """Verify ImputationFederate initialization and teardown."""
    config = ComponentDefinition(
        static_inputs=StaticInputs(name="test_imputer", deltat=0.1, end_time=10.0),
        dynamic_inputs=DynamicInputs(topology="feeder/topology", voltages_magnitude="sensor/voltages"),
        dynamic_outputs=DynamicOutputs(injections="imputed/injections"),
    )

    with (
        patch("imputation_federate.federate.create_federate_info") as mock_create_info,
        patch("helics.helicsCreateValueFederate") as mock_create_vfed,
        patch("imputation_federate.federate.register_subscription") as mock_reg_sub,
        patch("imputation_federate.federate.register_publication") as mock_reg_pub,
        patch("imputation_federate.federate.cleanup_federate") as mock_cleanup,
    ):
        mock_create_info.return_value = MagicMock()
        mock_create_vfed.return_value = "mock_vfed_handle"
        mock_reg_sub.return_value = "mock_sub_handle"
        mock_reg_pub.return_value = "mock_pub_handle"

        fed = ImputationFederate(config, broker_config=BrokerConfig())
        assert fed.vfed == "mock_vfed_handle"
        assert fed.sub.topology == "mock_sub_handle"
        assert fed.sub.voltages_magnitude == "mock_sub_handle"
        assert fed.pub.injections == "mock_pub_handle"

        fed.destroy()
        assert fed.vfed is None
        mock_cleanup.assert_called_once_with("mock_vfed_handle")
