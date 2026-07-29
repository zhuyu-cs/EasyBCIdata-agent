"""Platform registry for EasyBCI — WebUI + CLI only."""

from dataclasses import dataclass


@dataclass
class PlatformInfo:
    label: str
    default_toolset: str


PLATFORMS = {
    "api_server": PlatformInfo(label="WebUI", default_toolset="easybci-webui"),
    "cli": PlatformInfo(label="CLI", default_toolset="easybci-cli"),
}
