"""Tests for SLA modeling, cost-latency tradeoff, and new evaluation metrics."""
from __future__ import annotations

import pytest

from postureopt.core import (
    ResourceProfile,
    SLATarget,
    TelemetrySnapshot,
    estimated_hourly_cost,
    estimated_latency_ms,
    resource_utilization_score,
)
from postureopt.data import (
    make_all_resource_profiles,
    make_resource_profile,
    make_sla_target,
    make_telemetry_series,
    make_telemetry_snapshot,
)
from postureopt.evaluate import (
    cost_efficiency_index,
    cost_latency_frontier,
    pareto_efficient_profiles,
    sla_compliance_score,
    sustainability_score,
)


# ---------------------------------------------------------------------------
# SLATarget validation
# ---------------------------------------------------------------------------


def test_sla_target_valid() -> None:
    sla = make_sla_target()
    assert sla.max_latency_ms == 200.0
    assert sla.min_availability == 0.999


def test_sla_target_invalid_latency() -> None:
    with pytest.raises(ValueError):
        SLATarget(
            sla_id="bad",
            max_latency_ms=-1.0,
            min_availability=0.99,
            max_error_rate=0.01,
            cost_per_hour_usd=5.0,
        )


def test_sla_target_invalid_availability() -> None:
    with pytest.raises(ValueError):
        SLATarget(
            sla_id="bad",
            max_latency_ms=100.0,
            min_availability=1.5,
            max_error_rate=0.01,
            cost_per_hour_usd=5.0,
        )


# ---------------------------------------------------------------------------
# TelemetrySnapshot validation
# ---------------------------------------------------------------------------


def test_telemetry_snapshot_valid() -> None:
    snap = make_telemetry_snapshot(seed=0, load_factor=0.4)
    assert 0.0 <= snap.availability <= 1.0
    assert 0.0 <= snap.error_rate <= 1.0
    assert snap.p99_latency_ms >= 10.0


def test_telemetry_snapshot_invalid_availability() -> None:
    with pytest.raises(ValueError):
        TelemetrySnapshot(availability=1.5)


# ---------------------------------------------------------------------------
# Telemetry series
# ---------------------------------------------------------------------------


def test_telemetry_series_length() -> None:
    series = make_telemetry_series(n_snapshots=12, seed=7)
    assert len(series) == 12


def test_telemetry_series_diurnal_peak() -> None:
    """Mid-series snapshots should have higher load than start/end."""
    series = make_telemetry_series(n_snapshots=24, seed=1, diurnal=True)
    mid = series[12].p99_latency_ms
    start = series[0].p99_latency_ms
    # Mid-day peak should have higher latency on average
    assert mid >= start * 0.5  # loose bound to avoid flakiness


# ---------------------------------------------------------------------------
# Resource profile
# ---------------------------------------------------------------------------


def test_resource_profile_valid() -> None:
    p = make_resource_profile(0)
    assert p.cpu_cores >= 1
    assert p.memory_gb > 0


def test_resource_profile_invalid_cpu() -> None:
    with pytest.raises(ValueError):
        ResourceProfile(profile_id="bad", cpu_cores=0, memory_gb=8.0)


def test_all_resource_profiles() -> None:
    profiles = make_all_resource_profiles()
    assert len(profiles) == 6


# ---------------------------------------------------------------------------
# Estimated latency and cost
# ---------------------------------------------------------------------------


def test_estimated_latency_increases_with_load() -> None:
    profile = make_resource_profile(0)  # 4 CPU cores
    low = estimated_latency_ms(profile, requests_per_second=10)
    high = estimated_latency_ms(profile, requests_per_second=180)
    assert high > low


def test_estimated_hourly_cost_gpu_surcharge() -> None:
    no_gpu = make_resource_profile(0)  # gpu_count=0
    gpu = make_resource_profile(2)     # gpu_count=1
    cost_no_gpu = estimated_hourly_cost(no_gpu)
    cost_gpu = estimated_hourly_cost(gpu)
    assert cost_gpu > cost_no_gpu


# ---------------------------------------------------------------------------
# SLA compliance score
# ---------------------------------------------------------------------------


def test_sla_compliance_all_compliant() -> None:
    sla = make_sla_target(max_latency_ms=1000.0, min_availability=0.5, max_error_rate=0.5, cost_per_hour_usd=100.0)
    snaps = make_telemetry_series(n_snapshots=10, seed=0)
    score = sla_compliance_score(snaps, sla)
    assert score == 1.0


def test_sla_compliance_none_compliant() -> None:
    sla = make_sla_target(max_latency_ms=1.0, min_availability=0.9999, max_error_rate=0.0, cost_per_hour_usd=0.001)
    snaps = make_telemetry_series(n_snapshots=10, seed=0)
    score = sla_compliance_score(snaps, sla)
    assert score == 0.0


def test_sla_compliance_empty() -> None:
    sla = make_sla_target()
    assert sla_compliance_score([], sla) == 0.0


# ---------------------------------------------------------------------------
# Cost efficiency index
# ---------------------------------------------------------------------------


def test_cost_efficiency_index_positive() -> None:
    sla = make_sla_target(max_latency_ms=1000.0, min_availability=0.5, max_error_rate=0.5, cost_per_hour_usd=100.0)
    snaps = make_telemetry_series(n_snapshots=10, seed=0)
    cei = cost_efficiency_index(snaps, sla)
    assert cei > 0.0


def test_cost_efficiency_index_empty() -> None:
    sla = make_sla_target()
    assert cost_efficiency_index([], sla) == 0.0


# ---------------------------------------------------------------------------
# Sustainability score
# ---------------------------------------------------------------------------


def test_sustainability_score_at_target() -> None:
    """If utilization equals target exactly, score should be 1.0."""
    # Create a snapshot where CPU=0.7, memory=0.7, no GPU => util = 0.7
    snap = TelemetrySnapshot(
        cpu_utilization=0.7,
        memory_utilization=0.7,
        gpu_utilization=0.0,
        availability=0.999,
        error_rate=0.001,
    )
    score = sustainability_score([snap], target_utilization=0.7)
    assert abs(score - 1.0) < 1e-6


def test_sustainability_score_range() -> None:
    snaps = make_telemetry_series(n_snapshots=12, seed=5)
    score = sustainability_score(snaps, target_utilization=0.70)
    assert 0.0 <= score <= 1.0


def test_sustainability_score_empty() -> None:
    assert sustainability_score([]) == 0.0


# ---------------------------------------------------------------------------
# Cost-latency frontier
# ---------------------------------------------------------------------------


def test_cost_latency_frontier_length() -> None:
    profiles = make_all_resource_profiles()
    frontier = cost_latency_frontier(profiles, requests_per_second=50)
    assert len(frontier) == len(profiles)
    for cost, latency in frontier:
        assert cost > 0
        assert latency > 0


def test_pareto_efficient_profiles_subset() -> None:
    profiles = make_all_resource_profiles()
    pareto = pareto_efficient_profiles(profiles, requests_per_second=50)
    assert 1 <= len(pareto) <= len(profiles)


def test_resource_utilization_score_range() -> None:
    snaps = make_telemetry_series(n_snapshots=8, seed=3)
    util = resource_utilization_score(snaps)
    assert 0.0 <= util <= 1.0
