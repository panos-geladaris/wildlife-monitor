"""
Read environmental data from Breakout Garden I²C sensors.

Sensors:
- LTR-559: Ambient light (lux) and proximity
- BH1745: RGB + Clear colour channels
- BME688: Temperature, humidity, pressure, gas resistance
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LightReading:
    lux: float
    proximity: int


@dataclass
class ColourReading:
    red: int
    green: int
    blue: int
    clear: int


@dataclass
class EnvironmentReading:
    temperature: float  # °C
    humidity: float     # %
    pressure: float     # hPa
    gas_resistance: float  # Ohms


class SensorReader:
    """Reads from Breakout Garden I²C sensors. Each sensor is optional."""

    def __init__(self):
        self._ltr559 = None
        self._bh1745 = None
        self._bme688 = None
        self._init_sensors()

    def _init_sensors(self):
        # LTR-559
        try:
            from ltr559 import LTR559
            self._ltr559 = LTR559()
            logger.info("LTR-559 sensor initialised")
        except Exception as e:
            logger.warning(f"LTR-559 not available: {e}")

        # BH1745
        try:
            from bh1745 import BH1745
            self._bh1745 = BH1745()
            self._bh1745.setup()
            self._bh1745.set_leds(0)
            logger.info("BH1745 sensor initialised")
        except Exception as e:
            logger.warning(f"BH1745 not available: {e}")

        # BME688
        try:
            import bme680
            self._bme688 = bme680.BME680(bme680.I2C_ADDR_PRIMARY)
            self._bme688.set_humidity_oversample(bme680.OS_2X)
            self._bme688.set_pressure_oversample(bme680.OS_4X)
            self._bme688.set_temperature_oversample(bme680.OS_8X)
            self._bme688.set_filter(bme680.FILTER_SIZE_3)
            logger.info("BME688 sensor initialised")
        except Exception as e:
            logger.warning(f"BME688 not available: {e}")

    @property
    def available_sensors(self) -> list[str]:
        sensors = []
        if self._ltr559 is not None:
            sensors.append("ltr559")
        if self._bh1745 is not None:
            sensors.append("bh1745")
        if self._bme688 is not None:
            sensors.append("bme688")
        return sensors

    def read_light(self) -> LightReading | None:
        if self._ltr559 is None:
            return None
        try:
            lux = self._ltr559.get_lux()
            proximity = self._ltr559.get_proximity()
            return LightReading(lux=lux, proximity=proximity)
        except Exception as e:
            logger.error(f"LTR-559 read error: {e}")
            return None

    def read_colour(self) -> ColourReading | None:
        if self._bh1745 is None:
            return None
        try:
            r, g, b, c = self._bh1745.get_rgbc_raw()
            return ColourReading(red=r, green=g, blue=b, clear=c)
        except Exception as e:
            logger.error(f"BH1745 read error: {e}")
            return None

    def read_environment(self) -> EnvironmentReading | None:
        if self._bme688 is None:
            return None
        try:
            if not self._bme688.get_sensor_data():
                return None
            return EnvironmentReading(
                temperature=self._bme688.data.temperature,
                humidity=self._bme688.data.humidity,
                pressure=self._bme688.data.pressure,
                gas_resistance=self._bme688.data.gas_resistance,
            )
        except Exception as e:
            logger.error(f"BME688 read error: {e}")
            return None

    def read_all(self) -> dict:
        light = self.read_light()
        colour = self.read_colour()
        environment = self.read_environment()
        return {
            "light": {"lux": light.lux, "proximity": light.proximity} if light else None,
            "colour": {"red": colour.red, "green": colour.green, "blue": colour.blue, "clear": colour.clear} if colour else None,
            "environment": {"temperature": environment.temperature, "humidity": environment.humidity, "pressure": environment.pressure, "gas_resistance": environment.gas_resistance} if environment else None,
        }
