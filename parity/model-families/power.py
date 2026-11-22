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
    def read(supply, name):
        return (supply / name).read_text().strip() if (supply / name).is_file() else None

    # Mains, and the USB types a laptop charges through (USB, USB_C, USB_PD, ...), are external supplies.
    external = [s for s, kind in kinds.items() if kind == "Mains" or kind.startswith("USB")]
    if any(read(s, "online") == "1" for s in external):
        return True
    batteries = [s for s, kind in kinds.items() if kind == "Battery"]
    if not batteries:
        return True
    states = {read(s, "status") for s in batteries}
    if "Discharging" in states:
        return False
    if states & {"Charging", "Full"}:
        return True
    # A battery that does not say: every external supply reporting offline means battery; none reporting is unknown.
    reported = [read(s, "online") for s in external]
    return False if reported and all(state == "0" for state in reported) else None


def require_external_power(allow_battery, when="now"):
    if on_external_power() is False and not allow_battery:
        sys.exit(f"The machine is on battery power {when}. Connect external power, or pass --allow-battery.")
