"""Tests for postureopt.data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from postureopt.data import make_location, make_asset, make_posture_state, THEATER_LOCATIONS
from postureopt.core import Location, Asset


def test_make_location_type():
    loc = make_location(0)
    assert isinstance(loc, Location)


def test_make_asset_readiness_range():
    asset = make_asset(readiness=0.85)
    assert 0.0 <= asset.readiness_rate <= 1.0


def test_make_posture_state_n_assets():
    state = make_posture_state(n_assets=15, seed=1)
    assert len(state.assets) == 15


def test_theater_locations_length():
    assert len(THEATER_LOCATIONS) == 8
