from pathlib import Path

import pandas as pd
import great_expectations as gx

from app.config.db import get_db_connection


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DOCS_OUTPUT_DIR = (
    PROJECT_ROOT
    / "docs"
    / "evidence"
    / "sprint2"
    / "great_expectations_data_docs"
)

VALID_RANGES = {
    "soil_moisture": (0, 100),
    "soil_temp": (0, 60),
    "ambient_temp": (0, 60),
    "humidity": (0, 100),
    "pressure": (850, 1100),
    "solar_radiation": (0, 120000),
}


def load_sensor_data():
    conn = get_db_connection()

    query = """
        SELECT
            id,
            field_id,
            device_id,
            timestamp,
            parameter,
            value
        FROM sensor_readings
        ORDER BY timestamp ASC;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def validate_basic_columns(validator):
    validator.expect_column_values_to_not_be_null("id")
    validator.expect_column_values_to_not_be_null("field_id")
    validator.expect_column_values_to_not_be_null("device_id")
    validator.expect_column_values_to_not_be_null("timestamp")
    validator.expect_column_values_to_not_be_null("parameter")
    validator.expect_column_values_to_not_be_null("value")

    validator.expect_column_values_to_be_in_set(
        "parameter",
        list(VALID_RANGES.keys()),
    )


def validate_ranges_by_parameter(df, results):
    for parameter, (minimum_value, maximum_value) in VALID_RANGES.items():
        parameter_df = df[df["parameter"] == parameter].copy()

        if parameter_df.empty:
            results.append(
                {
                    "parameter": parameter,
                    "success": False,
                    "message": "No rows found for this parameter",
                    "invalid_count": None,
                }
            )
            continue

        invalid_rows = parameter_df[
            (parameter_df["value"] < minimum_value)
            | (parameter_df["value"] > maximum_value)
        ]

        results.append(
            {
                "parameter": parameter,
                "success": len(invalid_rows) == 0,
                "minimum_value": minimum_value,
                "maximum_value": maximum_value,
                "total_rows": len(parameter_df),
                "invalid_count": len(invalid_rows),
            }
        )


def create_html_report(basic_result, range_results):
    DATA_DOCS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    html_file = DATA_DOCS_OUTPUT_DIR / "sensor_readings_validation_report.html"

    basic_success = basic_result.success

    range_success = all(item["success"] for item in range_results)

    overall_success = basic_success and range_success

    rows_html = ""

    for item in range_results:
        rows_html += f"""
        <tr>
            <td>{item["parameter"]}</td>
            <td>{item["success"]}</td>
            <td>{item.get("minimum_value", "")}</td>
            <td>{item.get("maximum_value", "")}</td>
            <td>{item.get("total_rows", "")}</td>
            <td>{item.get("invalid_count", "")}</td>
        </tr>
        """

    html_content = f"""
    <html>
    <head>
        <title>Sensor Readings Great Expectations Report</title>
    </head>
    <body>
        <h1>Sensor Readings Great Expectations Report</h1>

        <h2>Overall Result</h2>
        <p><b>Success:</b> {overall_success}</p>

        <h2>Basic Column Validation</h2>
        <p><b>Success:</b> {basic_success}</p>

        <h2>Sensor Range Validation</h2>
        <table border="1" cellpadding="8" cellspacing="0">
            <tr>
                <th>Parameter</th>
                <th>Success</th>
                <th>Minimum</th>
                <th>Maximum</th>
                <th>Total Rows</th>
                <th>Invalid Count</th>
            </tr>
            {rows_html}
        </table>
    </body>
    </html>
    """

    html_file.write_text(html_content, encoding="utf-8")

    return html_file, overall_success


def main():
    print("Great Expectations sensor data validation")
    print("----------------------------------------")

    df = load_sensor_data()

    if df.empty:
        raise ValueError("No sensor readings found.")

    context = gx.get_context()

    data_source = context.data_sources.add_pandas("sensor_data_source")

    data_asset = data_source.add_dataframe_asset(name="sensor_readings_asset")

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "sensor_readings_batch"
    )

    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="sensor_readings_suite")

    validator = gx.validator.validator.Validator(
        execution_engine=gx.execution_engine.PandasExecutionEngine(),
        batches=[batch],
        expectation_suite=suite,
    )

    validate_basic_columns(validator)

    basic_result = validator.validate()

    range_results = []
    validate_ranges_by_parameter(df, range_results)

    html_file, overall_success = create_html_report(
        basic_result,
        range_results,
    )

    print(f"Rows validated: {len(df)}")
    print(f"Basic validation passed: {basic_result.success}")
    print(f"Report saved to: {html_file}")

    if not overall_success:
        raise ValueError("Great Expectations validation failed.")

    print("Great Expectations validation PASSED.")


if __name__ == "__main__":
    main()