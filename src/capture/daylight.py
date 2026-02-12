"""
Daylight-only capture gate using sunrise/sunset times.
"""

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


@dataclass
class SunTimes:
    sunrise: datetime
    sunset: datetime
    date: date


class SunriseSunsetClient:
    BASE_URL = "https://api.sunrise-sunset.org/json"

    def fetch(
        self, lat: float, lng: float, dt: date, tzid: str
    ) -> Optional[SunTimes]:
        params = (
            f"lat={lat}&lng={lng}"
            f"&date={dt.isoformat()}"
            f"&formatted=0"
            f"&tzid={tzid}"
        )
        url = f"{self.BASE_URL}?{params}"

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.error("Sunrise-sunset API request failed: %s", exc)
            return None

        try:
            results = data["results"]
            sunrise = datetime.fromisoformat(results["sunrise"])
            sunset = datetime.fromisoformat(results["sunset"])
        except (KeyError, ValueError) as exc:
            logger.error("Failed to parse sunrise-sunset response: %s", exc)
            return None

        return SunTimes(sunrise=sunrise, sunset=sunset, date=dt)


class DaylightGate:
    def __init__(
        self,
        client: SunriseSunsetClient,
        lat: float,
        lng: float,
        tzid: str,
        start_offset_minutes: int = 0,
        end_offset_minutes: int = 0,
        fallback: str = "allow",
        now_fn: Optional[Callable[[], datetime]] = None,
    ):
        self._client = client
        self._lat = lat
        self._lng = lng
        self._tzid = tzid
        self._start_offset = timedelta(minutes=start_offset_minutes)
        self._end_offset = timedelta(minutes=end_offset_minutes)
        self._fallback = fallback
        self._now_fn = now_fn or (lambda: datetime.now(ZoneInfo(tzid)))
        self._cached_sun_times: Optional[SunTimes] = None

    def refresh_if_needed(self, now: datetime) -> None:
        today = now.date()
        if self._cached_sun_times is not None and self._cached_sun_times.date == today:
            return

        sun_times = self._client.fetch(self._lat, self._lng, today, self._tzid)
        if sun_times is not None:
            self._cached_sun_times = sun_times
            effective_sunrise = sun_times.sunrise - self._start_offset
            effective_sunset = sun_times.sunset + self._end_offset
            logger.info(
                "Daylight window for %s: %s – %s",
                today.isoformat(),
                effective_sunrise.strftime("%H:%M"),
                effective_sunset.strftime("%H:%M"),
            )
        else:
            self._cached_sun_times = None
            logger.warning("Could not refresh sun times for %s", today.isoformat())

    def is_capture_allowed(self, now: Optional[datetime] = None) -> bool:
        if now is None:
            now = self._now_fn()

        self.refresh_if_needed(now)

        if self._cached_sun_times is None:
            return self._fallback == "allow"

        effective_sunrise = self._cached_sun_times.sunrise - self._start_offset
        effective_sunset = self._cached_sun_times.sunset + self._end_offset
        return effective_sunrise <= now <= effective_sunset

    def next_transition(
        self, now: Optional[datetime] = None
    ) -> Optional[tuple[datetime, bool]]:
        if now is None:
            now = self._now_fn()

        self.refresh_if_needed(now)

        if self._cached_sun_times is None:
            return None

        effective_sunrise = self._cached_sun_times.sunrise - self._start_offset
        effective_sunset = self._cached_sun_times.sunset + self._end_offset

        if now < effective_sunrise:
            return (effective_sunrise, True)
        if now <= effective_sunset:
            return (effective_sunset, False)

        tomorrow = now.date() + timedelta(days=1)
        tomorrow_sun = self._client.fetch(
            self._lat, self._lng, tomorrow, self._tzid
        )
        if tomorrow_sun is not None:
            return (tomorrow_sun.sunrise - self._start_offset, True)
        return None

    @property
    def sun_times(self) -> Optional[SunTimes]:
        return self._cached_sun_times
