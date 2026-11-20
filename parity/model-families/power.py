"""Report whether the machine runs on external power. A laptop on battery lowers its clocks, which invalidates timings."""
import ctypes
import os
import subprocess
import sys
from pathlib import Path


def on_external_power():
    """True on AC power, False on battery, None when the platform does not say. A desktop without a battery is AC."""
    if os.name == "nt":
        class SystemPowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        status = SystemPowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return None
        return {0: False, 1: True}.get(status.ACLineStatus)
    if sys.platform == "darwin":
        try:
            return "AC Power" in subprocess.check_output(["pmset", "-g", "batt"], text=True)
        except (OSError, subprocess.CalledProcessError):
            return None
    supplies = list(Path("/sys/class/power_supply").glob("*"))
    kinds = {s: (s / "type").read_text().strip() for s in supplies if (s / "type").is_file()}
    mains = [s for s, kind in kinds.items() if kind == "Mains"]
    if mains:
        return any((s / "online").read_text().strip() == "1" for s in mains if (s / "online").is_file())
    return None if "Battery" in kinds.values() else True


def require_external_power(allow_battery, when="now"):
    if on_external_power() is False and not allow_battery:
        sys.exit(f"The machine is on battery power {when}. Connect external power, or pass --allow-battery.")
