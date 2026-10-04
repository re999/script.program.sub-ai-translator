import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_XML = ROOT / "resources" / "settings.xml"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def settings_definitions():
    return {node.get("id"): node for node in ET.parse(SETTINGS_XML).getroot().iter("setting")}


class FakeAddon:
    stored = {}

    def __init__(self, addon_id=None):
        self.addon_id = addon_id

    def getSetting(self, setting_id):
        if setting_id in FakeAddon.stored:
            return FakeAddon.stored[setting_id]
        node = settings_definitions().get(setting_id)
        return node.get("default", "") if node is not None else ""

    def getSettingBool(self, setting_id):
        return self.getSetting(setting_id).lower() == "true"

    def getLocalizedString(self, string_id):
        return str(string_id)

    def getAddonInfo(self, key):
        return str(ROOT)


def install_fake_kodi_modules():
    xbmc = types.ModuleType("xbmc")
    xbmc.LOGDEBUG, xbmc.LOGINFO, xbmc.LOGERROR = 0, 1, 4
    xbmc.log = lambda message, level=0: None
    xbmcaddon = types.ModuleType("xbmcaddon")
    xbmcaddon.Addon = FakeAddon
    sys.modules.setdefault("xbmc", xbmc)
    sys.modules.setdefault("xbmcaddon", xbmcaddon)


install_fake_kodi_modules()
