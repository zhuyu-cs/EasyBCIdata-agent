"""Stub — skin_engine module removed (not needed for local BCI agent)."""


class SkinConfig:
    def __init__(self, *args, **kwargs):
        self.tool_prefix = "┊"
        self.tool_emojis = {}
        self.prompt_symbol = "> "
        self.goodbye = ""
        self.help_header = ""
        self.name = "default"

    def get(self, key, default=None):
        return getattr(self, key, default)

    # The skin engine is stubbed out (no active skin). These lookups always
    # return the caller's fallback so the "no skin configured" default renders.
    # Call sites live in cli.py / display.py / banner.py.
    def get_color(self, key, fallback=""):
        """Return a hex color for *key*; the stub has none, so fall back.

        Consumers expect a "#RRGGBB" string (they validate len==7 / leading
        '#') or an empty string that triggers their own fallback.
        """
        return fallback

    def get_branding(self, key, fallback=""):
        """Return a branding string (agent_name / response_label / welcome / …).
        The stub carries no branding overrides, so return the caller fallback."""
        return fallback

    def get_spinner_wings(self):
        """Decorative spinner side-glyphs. The stub has none."""
        return []


_DEFAULT_SKIN = SkinConfig()


def init_skin_from_config(*args, **kwargs):
    pass


def get_active_skin(*args, **kwargs):
    return _DEFAULT_SKIN


def get_active_skin_name(*args, **kwargs):
    return "default"


def get_active_goodbye(*args, **kwargs):
    return ""


def get_active_prompt_symbol(*args, **kwargs):
    return "> "


def get_active_help_header(*args, **kwargs):
    return ""


def get_prompt_toolkit_style_overrides(*args, **kwargs):
    return {}


def list_skins(*args, **kwargs):
    return ["default"]


def set_active_skin(*args, **kwargs):
    pass
