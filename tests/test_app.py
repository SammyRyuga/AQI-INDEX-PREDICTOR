"""Regression checks for the Codespaces Streamlit entry point."""

from io import StringIO
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "Model_Final.py"
CSV = "City,AQI,PM2.5,PM10\n" + "".join(
    f"{'Delhi' if i % 2 else 'Mumbai'},{40 + i * 3},{10 + i},{20 + i * 2}\n"
    for i in range(50)
)


def run_app(page="Home", csv=CSV):
    app = AppTest.from_file(str(APP), default_timeout=30)
    with patch("streamlit.file_uploader", side_effect=lambda *a, **kw: StringIO(csv)):
        app.run()
        if page != "Home":
            app.sidebar.radio[0].set_value(page).run()
    return app


def test_uploaded_dataset_renders_prediction_inputs():
    app = run_app()
    assert not app.exception
    assert len(app.number_input) == 2
    assert "PM2.5" in app.number_input[0].label
    assert "PM10" in app.number_input[1].label


def test_uploaded_dataset_renders_report_feature_analysis():
    app = run_app("View Model Report")
    assert not app.exception
    assert any("Available pollutant features" in item.value for item in app.markdown)
    assert any("PM2.5: 1.0000" in item.value for item in app.markdown)
    assert not any("Date range" in item.value for item in app.markdown)


def test_report_retains_date_range_when_present():
    lines = CSV.splitlines()
    csv = lines[0] + ",Date\n" + "\n".join(
        line + (",2025-01-01" if i % 2 else ",2025-01-31")
        for i, line in enumerate(lines[1:])
    )
    app = run_app("View Model Report", csv)
    assert not app.exception
    date_ranges = [item.value for item in app.markdown if "Date range" in item.value]
    assert date_ranges == ["**Date range**: 2025-01-01 to 2025-01-31"]


def test_uploaded_dataset_renders_graphs():
    app = run_app("Graphs")
    assert not app.exception
    assert len(app.get("imgs")) == 1


def test_upload_ignores_pollutant_with_no_observations():
    lines = CSV.splitlines()
    csv = lines[0] + ",NO2\n" + "\n".join(line + "," for line in lines[1:])
    app = run_app(csv=csv)
    assert not app.exception
    assert len(app.number_input) == 2
    assert any("NO2" in item.value for item in app.warning)


def test_upload_without_observed_pollutants_shows_error():
    csv = "City,AQI,PM2.5\n" + "".join(f"Delhi,{i},\n" for i in range(50))
    app = run_app(csv=csv)
    assert not app.exception
    assert any("No observed pollutant values" in item.value for item in app.error)
    assert not app.number_input


def test_pollutant_observations_require_a_valid_aqi():
    csv = "City,AQI,PM2.5,PM10\n" + "".join(
        f"Delhi,{i},,{i * 2}\n" for i in range(50)
    ) + "Delhi,,10,20\n"
    app = run_app(csv=csv)
    assert not app.exception
    assert len(app.number_input) == 1
    assert "PM10" in app.number_input[0].label
