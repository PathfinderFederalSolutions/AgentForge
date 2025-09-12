import pytest
from services.orchestrator.app.models import ThreatMarker, ThreatZone


def test_marker_bounds_and_sanitize():
    m = ThreatMarker(
        title="bad\x00title" * 20,
        description="desc\n" * 400,
        coordinates=(190.0, 95.0),
        severity="high",
        evidence_id="ev\x07id",
    )
    lon, lat = m.coordinates
    assert -180.0 <= lon <= 180.0
    assert -90.0 <= lat <= 90.0
    assert m.title is not None and "\x00" not in m.title and len(m.title) <= 256
    assert m.description is not None and "\n" not in m.description and len(m.description) <= 1024
    assert m.evidence_id is not None and "\x07" not in m.evidence_id


def test_zone_ring_closure_and_bounds():
    z = ThreatZone(
        title="Z",
        polygon=[[(181, 45), (181, 46), (182, 46), (181, 45)]],
    )
    ring = z.polygon[0]
    assert ring[0] == ring[-1]
    for lon, lat in ring:
        assert -180.0 <= lon <= 180.0
        assert -90.0 <= lat <= 90.0
