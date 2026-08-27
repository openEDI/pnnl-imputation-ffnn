"""FastAPI server for the OEDISI Imputation Federate."""

import json
import logging
import os
import socket
import traceback

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from oedisi.componentframework.system_configuration import ComponentStruct
from oedisi.types.common import BrokerConfig, DefaultFileNames, HealthCheck, ServerReply

from .federate import run_simulator
from .schemas import StaticInputs

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

app = FastAPI(title="OEDISI Imputation Federate Server")


@app.get("/")
def read_root() -> JSONResponse:
    """Health check endpoint."""
    hostname = socket.gethostname()
    host_ip = "127.0.0.1"
    try:
        host_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        try:
            host_ip = socket.gethostbyname(hostname + ".local")
        except socket.gaierror:
            pass

    response = HealthCheck(hostname=hostname, host_ip=host_ip).model_dump()
    return JSONResponse(response, 200)


@app.post("/configure")
async def configure(component_struct: ComponentStruct) -> JSONResponse:
    """Validate and write federate configuration files."""
    try:
        component = component_struct.component
        params = dict(component.parameters)
        params["name"] = component.name

        # Validate static inputs schema
        StaticInputs.model_validate(params)

        links = {}
        for link in component_struct.links:
            links[link.target_port] = f"{link.source}/{link.source_port}"

        with open(DefaultFileNames.INPUT_MAPPING.value, "w", encoding="utf-8") as f:
            json.dump(links, f, indent=2)

        with open(DefaultFileNames.STATIC_INPUTS.value, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        response = ServerReply(detail="Successfully updated configuration files.").model_dump()
        return JSONResponse(response, 200)
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}") from e


@app.post("/run")
async def run_model(broker_config: BrokerConfig, background_tasks: BackgroundTasks) -> JSONResponse:
    """Accept broker config and launch simulation background task."""
    logger.info(f"Received broker configuration: {broker_config}")
    try:
        background_tasks.add_task(run_simulator, broker_config)
        response = ServerReply(detail="Task successfully added.").model_dump()
        return JSONResponse(response, 200)
    except Exception as e:
        err = traceback.format_exc()
        logger.error(f"Failed to start simulator: {err}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def main() -> None:
    """Entry point for imputation-server console script."""
    port = int(os.environ.get("PORT", "5905"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
