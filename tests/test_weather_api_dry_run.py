"""CLI smoke tests read tiny query files and never call weather services."""

import json

import pytest

from flight_delay.weather.api_dry_run import main


def test_cli_uses_same_queries_deduplicates_and_preserves_mapping(tmp_path, capsys):
    path = tmp_path / "queries.json"
    rows = [{"Origin": "ATL", "scheduled_departure_utc": "2023-07-15T09:15Z"}] * 2
    path.write_text(json.dumps(rows))
    assert main(["--input", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry_run"
    assert report["input_queries"] == 2
    assert report["unique_http_requests"] == 1
    assert len(report["query_mapping"]) == 2
    assert (
        report["query_mapping"][0]["request_key"]
        == report["query_mapping"][1]["request_key"]
    )
    assert "outside_documented_ifs_archive" in report["requests"][0]["blockers"]
    assert report["http_attempts"] == 0


def test_cli_reports_over_budget_with_nonzero_status(tmp_path, capsys):
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            [
                {"Origin": origin, "scheduled_departure_utc": "2024-07-15T09:15Z"}
                for origin in ["ATL", "SFO"]
            ]
        )
    )
    assert main(["--input", str(path), "--max-requests", "1"]) == 2
    assert not json.loads(capsys.readouterr().out)["within_request_budget"]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        {},
        [{"Origin": "ATL", "DepDel15": 1}],
        [
            {
                "Origin": "ATL",
                "scheduled_departure_utc": "2025-01-01T00:00Z",
                "ArrDel15": 0,
            }
        ],
        [{"Origin": "ATL", "scheduled_departure_utc": "2024-01-01T00:00Z"}] * 101,
    ],
)
def test_cli_rejects_wrong_shape_labels_and_large_inputs(tmp_path, rows):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(SystemExit):
        main(["--input", str(path)])


def test_cli_has_no_live_or_paid_switch():
    with pytest.raises(SystemExit):
        main(["--input", "unused", "--live"])
