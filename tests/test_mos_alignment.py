"""Synthetic full-run receipts validate safe alignment, not real API availability."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from flight_delay.weather.flight_times import convert_bts_departure_to_utc
from flight_delay.weather.iem_mos import MOSContractError
from flight_delay.weather.mos_alignment import (
    MODEL_FIELDS,
    RunAvailabilityEvidence,
    align_mos_forecasts,
    require_model_ready_alignment,
)
from flight_delay.weather.mos_contracts import CONTRACT_VERSION, FEATURE_COLUMNS


def weather(times=("2024-07-16T09:00Z", "2024-07-16T12:00Z"), model="GFS"):
    run = pd.Timestamp("2024-07-15T00:00Z" if model == "GFS" else "2024-07-15T01:00Z")
    rows = []
    for i, time in enumerate(times):
        valid = pd.Timestamp(time)
        r = {f: np.nan for f in FEATURE_COLUMNS}
        r.update(
            Origin="ATL",
            model=model,
            provider="iem_mos",
            backend="iem_mos",
            contract_version=CONTRACT_VERSION,
            forecast_run_at_utc=run,
            forecast_valid_at_utc=valid,
            forecast_hour=(valid - run).total_seconds() / 3600,
            response_sha256="a" * 64,
            request_key="b" * 64,
            mos_temperature_c=10.0 + 6 * i,
            mos_wind_speed_mps=2.0 + 6 * i,
            mos_gfs_visibility_category=2.0 + 4 * i,
            mos_gfs_ceiling_category=2.0 + 4 * i,
            mos_pop_6h_pct=10.0 + 80 * i,
            mos_tstorm_6h_pct=20.0 + 70 * i,
            mos_gust_mps=5.0 + 10 * i,
            mos_nbm_visibility_miles=2.0 + 8 * i,
            mos_nbm_ceiling_hundreds_ft=10.0 + 90 * i,
            mos_nbm_ceiling_unlimited=0.0,
            mos_tstorm_3h_pct=10.0 + 80 * i,
        )
        rows.append(r)
    return pd.DataFrame(rows)


def flights(w, target="2024-07-16T10:00Z"):
    return pd.DataFrame(
        {
            "Origin": ["ATL"],
            "model": [w.model.iloc[0]],
            "forecast_run_at_utc": [w.forecast_run_at_utc.iloc[0]],
            "scheduled_departure_utc": [pd.Timestamp(target)],
            "sample_id": ["fixture"],
        }
    )


def evidence(w, **changes):
    # Explicit synthetic fixture. No real forecast receives this certification.
    e = RunAvailabilityEvidence(
        w.model.iloc[0],
        "KATL",
        w.forecast_run_at_utc.iloc[0],
        "a" * 64,
        "historical_full_run_receipt",
        w.forecast_run_at_utc.iloc[0] + pd.Timedelta(hours=4),
        "synthetic-historical-full-run-receipt",
        True,
    )
    return replace(e, **changes)


def test_conservative_keeps_aged_point_and_does_not_shift_probability_window():
    w = weather()
    f = flights(w)
    r = require_model_ready_alignment(f, w, evidence=[evidence(w)]).iloc[0]
    assert r.temporal_match and r.strict_pit_pass
    assert r.valid_time_gap_hours == 1
    assert r.mos_temperature_c == 10
    assert r.mos_gfs_visibility_category == 2
    assert pd.isna(r.mos_pop_6h_pct) and pd.isna(r.mos_tstorm_6h_pct)
    assert r.forecast_age_at_prediction_hours == 10


def test_linear_interpolates_only_allowlisted_continuous_values():
    w = weather()
    r = require_model_ready_alignment(
        flights(w), w, strategy="interpolation", evidence=[evidence(w)]
    ).iloc[0]
    assert r.mos_temperature_c == pytest.approx(12)
    assert r.mos_wind_speed_mps == pytest.approx(4)
    assert r.mos_gfs_visibility_category == 2 and r.mos_gfs_ceiling_category == 2
    assert pd.isna(r.mos_pop_6h_pct)
    assert r.right_weight == pytest.approx(1 / 3)
    assert r.bracket_span_hours == 3 and r.interpolation_executed


def test_nbm_visibility_ceiling_gust_and_probabilities_are_not_blended():
    w = weather(model="NBS")
    r = align_mos_forecasts(
        flights(w), w, strategy="interpolation", evidence=[evidence(w)]
    ).iloc[0]
    assert r.mos_nbm_visibility_miles == 2
    assert r.mos_nbm_ceiling_hundreds_ft == 10
    assert r.mos_gust_mps == 5
    assert pd.isna(r.mos_tstorm_3h_pct)


@pytest.mark.parametrize("strategy", ["conservative", "interpolation"])
def test_unknown_publication_never_materializes_features_or_passes(strategy):
    w = weather()
    r = align_mos_forecasts(flights(w), w, strategy=strategy).iloc[0]
    assert r.temporal_match and r.potential_field_count > 0
    assert r.availability_status == "availability_unverified"
    assert not r.model_ready and not r.strict_pit_pass and not r.interpolation_executed
    assert r.materialized_field_count == 0
    assert all(pd.isna(r[f]) for f in FEATURE_COLUMNS)
    with pytest.raises(MOSContractError, match="not model-ready"):
        require_model_ready_alignment(flights(w), w, strategy=strategy)


def test_runtime_and_todays_retrieval_flags_cannot_create_release_evidence():
    w = weather()
    w["availability_status"] = "availability_verified"
    w["published_at_utc"] = w.forecast_run_at_utc
    w["retrieved_at_utc"] = pd.Timestamp("2026-11-21T00:00Z")
    assert not align_mos_forecasts(flights(w), w).strict_pit_pass.any()


def test_schedule_lag_is_disclosed_but_cannot_pass_audit():
    w = weather()
    e = evidence(w, kind="schedule_assumption", covers_entire_station_run=False)
    r = align_mos_forecasts(flights(w), w, evidence=[e]).iloc[0]
    assert r.assumed_available_time_utc == e.available_at
    assert pd.isna(r.actual_publication_time_utc)
    assert not r.strict_pit_pass


def test_historical_receipt_is_available_by_not_actual_publication():
    w = weather()
    r = align_mos_forecasts(flights(w), w, evidence=[evidence(w)]).iloc[0]
    assert r.strict_pit_pass and pd.isna(r.actual_publication_time_utc)
    assert pd.notna(r.verified_available_by_utc)


@pytest.mark.parametrize("offset,passed", [(0, True), (1, False)])
def test_publication_boundary_is_exact_to_the_second(offset, passed):
    w = weather()
    f = flights(w)
    cutoff = f.scheduled_departure_utc.iloc[0] - pd.Timedelta(hours=24)
    e = evidence(
        w,
        kind="verified_publication",
        available_at=cutoff + pd.Timedelta(seconds=offset),
    )
    r = align_mos_forecasts(f, w, strategy="interpolation", evidence=[e]).iloc[0]
    assert bool(r.strict_pit_pass) == passed
    assert bool(r.interpolation_executed) == passed
    assert r.actual_publication_time_utc == e.available_at


@pytest.mark.parametrize(
    "change",
    [
        {"available_at": "2024-07-14T23:59Z"},
        {"available_at": "2024-07-15T04:00"},
        {"evidence_reference": None},
        {"covers_entire_station_run": False},
        {"covers_entire_station_run": "true"},
        {"response_sha256": "bad"},
        {"kind": "model_run_time"},
    ],
)
def test_invalid_or_partial_evidence_is_not_certification(change):
    with pytest.raises(ValueError):
        evidence(weather(), **change)


def test_evidence_is_bound_to_original_payload_and_cannot_conflict():
    w = weather()
    with pytest.raises(MOSContractError, match="different payload"):
        align_mos_forecasts(
            flights(w), w, evidence=[evidence(w, response_sha256="c" * 64)]
        )
    with pytest.raises(MOSContractError, match="Conflicting"):
        align_mos_forecasts(flights(w), w, evidence=[evidence(w), evidence(w)])


def test_cross_midnight_interpolation_uses_utc_elapsed_time():
    w = weather(("2024-07-15T23:00Z", "2024-07-16T02:00Z"))
    f = flights(w, "2024-07-16T00:00Z")
    # Receipt at init+4h is after this cutoff: explicitly use a valid earlier
    # full-run fixture receipt to isolate the cross-day arithmetic.
    e = evidence(w, available_at=w.forecast_run_at_utc.iloc[0])
    r = align_mos_forecasts(f, w, strategy="interpolation", evidence=[e]).iloc[0]
    assert r.mos_temperature_c == pytest.approx(12)


def test_spring_dst_target_is_converted_to_utc_without_changing_24h_cutoff():
    w = weather(("2024-03-10T09:00Z", "2024-03-10T12:00Z"))
    run = pd.Timestamp("2024-03-09T00:00Z")
    w["forecast_run_at_utc"] = run
    w["forecast_hour"] = (w.forecast_valid_at_utc - run).dt.total_seconds() / 3600
    f = flights(w, "2024-03-10T03:30:00-07:00")
    r = align_mos_forecasts(
        f, w, strategy="interpolation", evidence=[evidence(w)]
    ).iloc[0]
    assert r.scheduled_departure_utc == pd.Timestamp("2024-03-10T10:30Z")
    assert r.flight_prediction_time_utc == pd.Timestamp("2024-03-09T10:30Z")
    assert r.right_weight == 0.5


def test_ambiguous_and_nonexistent_local_bts_times_remain_rejected():
    for date, clock in [("2024-11-03", 130), ("2024-03-10", 230)]:
        frame = pd.DataFrame(
            {"Origin": ["SFO"], "FlightDate": [date], "CRSDepTime": [clock]}
        )
        converted = convert_bts_departure_to_utc(frame).flights
        assert converted.departure_time_status.iloc[0] == "dst_unresolvable"
        w = weather()
        f = flights(w)
        f["scheduled_departure_utc"] = converted.scheduled_departure_utc
        with pytest.raises(ValueError):
            align_mos_forecasts(f, w)


@pytest.mark.parametrize("side", [0, 1])
def test_missing_endpoint_is_not_imputed_or_silently_replaced(side):
    w = weather()
    w.loc[side, "mos_temperature_c"] = np.nan
    r = align_mos_forecasts(
        flights(w), w, strategy="interpolation", evidence=[evidence(w)]
    ).iloc[0]
    assert pd.isna(r.mos_temperature_c) and not r.model_ready
    assert r.strict_pit_pass  # timing approval is separate from field completeness


def test_different_run_cannot_supply_the_right_interpolation_endpoint():
    w = weather()
    run = w.forecast_run_at_utc.iloc[0] + pd.Timedelta(hours=6)
    w.loc[1, "forecast_run_at_utc"] = run
    w.loc[1, "forecast_hour"] -= 6
    r = align_mos_forecasts(flights(w), w, strategy="interpolation").iloc[0]
    assert not r.temporal_match and r.alignment_status == "no_same_run_bracket"


@pytest.mark.parametrize(
    "change", ["duplicate", "payload", "observation", "lead", "naive"]
)
def test_corrupt_or_mixed_sources_fail_closed(change):
    w = weather()
    f = flights(w)
    if change == "duplicate":
        w = pd.concat([w, w.iloc[[0]]])
    elif change == "payload":
        w.loc[1, "response_sha256"] = "c" * 64
    elif change == "observation":
        w["provider"] = "asos_observations"
    elif change == "lead":
        w["forecast_hour"] += 1
    else:
        w["forecast_valid_at_utc"] = w.forecast_valid_at_utc.dt.tz_localize(None)
    with pytest.raises(ValueError):
        align_mos_forecasts(f, w)


@pytest.mark.parametrize("target", ["2024-07-16T08:59Z", "2024-07-16T13:00Z"])
def test_interpolation_does_not_extrapolate(target):
    w = weather()
    r = align_mos_forecasts(flights(w, target), w, strategy="interpolation").iloc[0]
    assert not r.temporal_match


def test_gap_caps_are_enforced_without_tuning_to_cover_all_targets():
    w = weather(("2024-07-16T06:00Z", "2024-07-16T12:00Z"))
    r = align_mos_forecasts(
        flights(w, "2024-07-16T07:00Z"), w, strategy="interpolation"
    ).iloc[0]
    assert not r.temporal_match and r.alignment_status == "bracket_gap_exceeded"
    assert r.right_gap_hours == 5


def test_exact_point_does_not_need_a_future_bracket_and_retains_period_endpoint():
    w = weather().iloc[[0]].copy()
    f = flights(w, "2024-07-16T09:00Z")
    r = align_mos_forecasts(
        f, w, strategy="interpolation", evidence=[evidence(w)]
    ).iloc[0]
    assert r.model_ready and not r.interpolation_executed
    assert r.mos_pop_6h_pct == 10


def test_future_valid_endpoint_is_allowed_only_when_whole_run_was_available():
    w = weather()
    f = flights(w)
    assert w.forecast_valid_at_utc.max() > f.scheduled_departure_utc.iloc[0]
    assert align_mos_forecasts(
        f, w, strategy="interpolation", evidence=[evidence(w)]
    ).strict_pit_pass.all()


def test_wrong_cutoff_and_unbounded_policy_are_rejected():
    w = weather()
    f = flights(w)
    f["prediction_at_utc"] = pd.Timestamp("2024-07-15T11:00Z")
    with pytest.raises(MOSContractError, match="T-24h"):
        align_mos_forecasts(f, w)
    for gap in (float("inf"), 0, True, 25):
        with pytest.raises(MOSContractError):
            align_mos_forecasts(flights(w), w, max_past_gap_hours=gap)


def test_flight_order_duplicates_and_inputs_are_preserved():
    w = weather()
    original = w.copy(deep=True)
    f = pd.concat([flights(w), flights(w)], ignore_index=True)
    r = align_mos_forecasts(f, w, evidence=[evidence(w)])
    assert len(r) == 2 and r.input_row_position.tolist() == [0, 1]
    pd.testing.assert_frame_equal(w, original)
    assert r.potential_field_denominator.tolist() == [len(MODEL_FIELDS["GFS"])] * 2


def test_dst_fall_back_explicit_offsets_represent_distinct_instants():
    w = weather(("2024-11-03T06:00Z", "2024-11-03T09:00Z", "2024-11-03T12:00Z"))
    run = pd.Timestamp("2024-11-02T00:00Z")
    w["forecast_run_at_utc"] = run
    w["forecast_hour"] = (w.forecast_valid_at_utc - run).dt.total_seconds() / 3600
    a = align_mos_forecasts(
        flights(w, "2024-11-03T01:30:00-07:00"),
        w,
        strategy="interpolation",
        evidence=[evidence(w)],
    ).iloc[0]
    b = align_mos_forecasts(
        flights(w, "2024-11-03T01:30:00-08:00"),
        w,
        strategy="interpolation",
        evidence=[evidence(w)],
    ).iloc[0]
    assert b.scheduled_departure_utc - a.scheduled_departure_utc == pd.Timedelta(
        hours=1
    )
    assert b.flight_prediction_time_utc - a.flight_prediction_time_utc == pd.Timedelta(
        hours=1
    )


def test_future_initialization_cannot_pass_from_temporal_matching_alone():
    w = weather()
    f = flights(w)
    run = pd.Timestamp("2024-07-15T12:00Z")
    w["forecast_run_at_utc"] = run
    w["forecast_hour"] = (w.forecast_valid_at_utc - run).dt.total_seconds() / 3600
    f["forecast_run_at_utc"] = run
    r = align_mos_forecasts(f, w, strategy="interpolation").iloc[0]
    assert r.temporal_match and not r.strict_pit_pass
    assert r.availability_status == "forecast_initialized_after_cutoff"


def test_actual_stage2_offline_replay_has_no_network_or_model_ready_rows(
    tmp_path, monkeypatch
):
    import importlib.util
    from pathlib import Path

    from flight_delay.weather.iem_mos import IEMMOSClient

    def forbidden(*args, **kwargs):
        pytest.fail("Offline replay must not acquire any data")

    monkeypatch.setattr(IEMMOSClient, "fetch", forbidden)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "alignment_replay", root / "scripts/replay_mos_alignment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.replay(root / "artifacts/iem_mos_pilot", tmp_path)
    assert summary["new_mos_json_requests"] == 0
    assert summary["forecast_records"] == 1584
    assert summary["flight_model_checks_per_strategy"] == 72
    for row in summary["comparison"]:
        assert row["temporal_matches"] == 72
        assert row["strict_pit_passes"] == row["materialized_field_values"] == 0
