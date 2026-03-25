"""
Configuration loading for ARIA.

Exposes the `AriaConfig` data model and `load_config` helper
for reading `aria_config.yaml` with environment-variable expansion.
"""

from .loader import AriaConfig as AriaConfig, load_config as load_config
