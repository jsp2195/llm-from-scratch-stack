"""Configuration helpers."""

from hydra.core.config_store import ConfigStore

from llmstack.config.schema import RootConfig


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=RootConfig)
