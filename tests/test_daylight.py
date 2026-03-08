"""Tests for the daylight-only capture module."""

import json
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.capture.daylight import DaylightGate, SunriseSunsetClient, SunTimes

TZ = ZoneInfo("Europe/London")

SAMPLE_RESPONSE = {
    "results": {
        "sunrise": "2025-06-15T05:00:00+01:00",
        "sunset": "2025-06-15T21:15:00+01:00",
        "solar_noon": "2025-06-15T13:07:30+01:00",
        "day_length": 58500,
    },
    "status": "OK",
}

SUNRISE = datetime.fromisoformat("2025-06-15T05:00:00+01:00")
SUNSET = datetime.fromisoformat("2025-06-15T21:15:00+01:00")


class TestSunriseSunsetClient:
    def test_fetch_success(self):
        body = json.dumps(SAMPLE_RESPONSE).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            client = SunriseSunsetClient()
            result = client.fetch(51.5, -0.16, date(2025, 6, 15), "Europe/London")

        assert result is not None
        assert result.sunrise == SUNRISE
        assert result.sunset == SUNSET
        assert result.date == date(2025, 6, 15)
        mock_open.assert_called_once()
        req = mock_open.call_args[0][0]
        url = req.full_url
        assert "formatted=0" in url
        assert "tzid=Europe/London" in url

    def test_fetch_network_error(self):
        with patch("urllib.request.urlopen", side_effect=OSError("no network")):
            client = SunriseSunsetClient()
            result = client.fetch(51.5, -0.16, date(2025, 6, 15), "Europe/London")
        assert result is None

    def test_fetch_bad_json(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            client = SunriseSunsetClient()
            result = client.fetch(51.5, -0.16, date(2025, 6, 15), "Europe/London")
        assert result is None

    def test_fetch_missing_keys(self):
        body = json.dumps({"results": {}, "status": "OK"}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            client = SunriseSunsetClient()
            result = client.fetch(51.5, -0.16, date(2025, 6, 15), "Europe/London")
        assert result is None


class TestDaylightGate:
    def _make_gate(self, now_time, sun_times=None, fallback="allow", start_offset=0, end_offset=0):
        client = MagicMock(spec=SunriseSunsetClient)
        if sun_times is not None:
            client.fetch.return_value = sun_times
        else:
            client.fetch.return_value = SunTimes(
                sunrise=SUNRISE, sunset=SUNSET, date=now_time.date()
            )
        gate = DaylightGate(
            client=client,
            lat=51.5,
            lng=-0.16,
            tzid="Europe/London",
            start_offset_minutes=start_offset,
            end_offset_minutes=end_offset,
            fallback=fallback,
            now_fn=lambda: now_time,
        )
        return gate, client

    def test_allowed_during_daytime(self):
        midday = datetime(2025, 6, 15, 12, 0, tzinfo=TZ)
        gate, _ = self._make_gate(midday)
        assert gate.is_capture_allowed() is True

    def test_blocked_before_sunrise(self):
        early = datetime(2025, 6, 15, 3, 0, tzinfo=TZ)
        gate, _ = self._make_gate(early)
        assert gate.is_capture_allowed() is False

    def test_blocked_after_sunset(self):
        late = datetime(2025, 6, 15, 23, 0, tzinfo=TZ)
        gate, _ = self._make_gate(late)
        assert gate.is_capture_allowed() is False

    def test_allowed_at_sunrise(self):
        gate, _ = self._make_gate(SUNRISE)
        assert gate.is_capture_allowed() is True

    def test_allowed_at_sunset(self):
        gate, _ = self._make_gate(SUNSET)
        assert gate.is_capture_allowed() is True

    def test_start_offset_extends_before_sunrise(self):
        before_sunrise = SUNRISE - timedelta(minutes=15)
        gate, _ = self._make_gate(before_sunrise, start_offset=30)
        assert gate.is_capture_allowed() is True

    def test_end_offset_extends_after_sunset(self):
        after_sunset = SUNSET + timedelta(minutes=15)
        gate, _ = self._make_gate(after_sunset, end_offset=30)
        assert gate.is_capture_allowed() is True

    def test_fallback_allow_on_api_failure(self):
        night = datetime(2025, 6, 15, 3, 0, tzinfo=TZ)
        gate, _ = self._make_gate(night, sun_times=None, fallback="allow")
        gate._daylight_gate = None
        client = MagicMock(spec=SunriseSunsetClient)
        client.fetch.return_value = None
        gate._client = client
        gate._cached_sun_times = None
        assert gate.is_capture_allowed() is True

    def test_fallback_block_on_api_failure(self):
        night = datetime(2025, 6, 15, 3, 0, tzinfo=TZ)
        client = MagicMock(spec=SunriseSunsetClient)
        client.fetch.return_value = None
        gate = DaylightGate(
            client=client,
            lat=51.5,
            lng=-0.16,
            tzid="Europe/London",
            fallback="block",
            now_fn=lambda: night,
        )
        assert gate.is_capture_allowed() is False

    def test_cache_reuses_same_day(self):
        midday = datetime(2025, 6, 15, 12, 0, tzinfo=TZ)
        gate, client = self._make_gate(midday)
        gate.is_capture_allowed()
        gate.is_capture_allowed()
        assert client.fetch.call_count == 1

    def test_cache_refreshes_on_new_day(self):
        day1 = datetime(2025, 6, 15, 12, 0, tzinfo=TZ)
        gate, client = self._make_gate(day1)
        gate.is_capture_allowed()

        day2 = datetime(2025, 6, 16, 12, 0, tzinfo=TZ)
        day2_sun = SunTimes(
            sunrise=datetime(2025, 6, 16, 4, 58, tzinfo=TZ),
            sunset=datetime(2025, 6, 16, 21, 16, tzinfo=TZ),
            date=date(2025, 6, 16),
        )
        client.fetch.return_value = day2_sun
        gate._now_fn = lambda: day2
        gate.is_capture_allowed(day2)
        assert client.fetch.call_count == 2

    def test_next_transition_before_sunrise(self):
        early = datetime(2025, 6, 15, 3, 0, tzinfo=TZ)
        gate, _ = self._make_gate(early)
        transition = gate.next_transition()
        assert transition is not None
        assert transition[0] == SUNRISE
        assert transition[1] is True

    def test_next_transition_during_day(self):
        midday = datetime(2025, 6, 15, 12, 0, tzinfo=TZ)
        gate, _ = self._make_gate(midday)
        transition = gate.next_transition()
        assert transition is not None
        assert transition[0] == SUNSET
        assert transition[1] is False

    def test_next_transition_after_sunset_returns_tomorrow_sunrise(self):
        late = datetime(2025, 6, 15, 23, 0, tzinfo=TZ)
        gate, client = self._make_gate(late)
        tomorrow_sun = SunTimes(
            sunrise=datetime(2025, 6, 16, 4, 58, tzinfo=TZ),
            sunset=datetime(2025, 6, 16, 21, 16, tzinfo=TZ),
            date=date(2025, 6, 16),
        )
        client.fetch.return_value = tomorrow_sun
        transition = gate.next_transition()
        assert transition is not None
        assert transition[0] == tomorrow_sun.sunrise
        assert transition[1] is True

    def test_next_transition_after_sunset_api_failure(self):
        late = datetime(2025, 6, 15, 23, 0, tzinfo=TZ)
        gate, client = self._make_gate(late)
        client.fetch.return_value = None
        gate._cached_sun_times = SunTimes(
            sunrise=SUNRISE, sunset=SUNSET, date=date(2025, 6, 15)
        )
        transition = gate.next_transition(late)
        assert transition is None

    def test_next_transition_with_offsets(self):
        before_offset_sunrise = SUNRISE - timedelta(minutes=40)
        gate, _ = self._make_gate(before_offset_sunrise, start_offset=30)
        transition = gate.next_transition()
        expected = SUNRISE - timedelta(minutes=30)
        assert transition is not None
        assert transition[0] == expected
        assert transition[1] is True

    def test_stale_cache_cleared_on_api_failure(self):
        day1 = datetime(2025, 6, 15, 12, 0, tzinfo=TZ)
        gate, client = self._make_gate(day1)
        gate.is_capture_allowed()
        assert gate.sun_times is not None

        day2 = datetime(2025, 6, 16, 12, 0, tzinfo=TZ)
        client.fetch.return_value = None
        gate._now_fn = lambda: day2
        result = gate.is_capture_allowed(day2)
        assert gate.sun_times is None
        assert result is True  # fallback="allow"

    def test_sun_times_property(self):
        midday = datetime(2025, 6, 15, 12, 0, tzinfo=TZ)
        gate, _ = self._make_gate(midday)
        assert gate.sun_times is None
        gate.is_capture_allowed()
        assert gate.sun_times is not None
        assert gate.sun_times.sunrise == SUNRISE
