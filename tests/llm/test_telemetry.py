"""A cost log is only useful if it is honest about what it does not know."""

from datetime import UTC, datetime

from agentic_erp_assistant.llm.telemetry import (
    InMemoryTelemetry,
    ModelCallRecord,
    TelemetrySink,
    now,
    price,
)


def make_record(**overrides) -> ModelCallRecord:
    fields = {
        "model": "gpt-4o",
        "outcome": "answered",
        "estimated_input_tokens": 800,
        "input_tokens": 812,
        "output_tokens": 57,
        "cost_usd": 0.00273,
        "latency_seconds": 1.5,
        "attempts": 1,
        "occurred_at": datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
    }
    fields.update(overrides)
    return ModelCallRecord(**fields)


def test_a_record_keeps_the_estimate_and_the_real_count_apart() -> None:
    """Two numbers on purpose: the gap is how estimate drift becomes visible."""
    record = make_record(estimated_input_tokens=800, input_tokens=812)

    assert record.estimated_input_tokens == 800
    assert record.input_tokens == 812
    assert record.estimate_error == -12


def test_a_record_that_never_reached_the_provider_has_no_estimate_error() -> None:
    record = make_record(outcome="budget_exceeded", input_tokens=0, output_tokens=0)

    assert record.estimate_error is None


def test_a_record_cannot_be_edited_afterwards() -> None:
    record = make_record()

    try:
        record.cost_usd = 0.0  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("a record describing the past should be frozen")


def test_a_priced_model_is_priced_from_the_reviewed_table() -> None:
    cost, detail = price("gpt-4o", input_tokens=1_000_000, output_tokens=0)

    assert cost == 2.50
    assert detail is None


def test_input_and_output_are_priced_separately() -> None:
    """Output bills at four times input on this model; a blended rate would be
    wrong by however talkative the model happened to be."""
    cost, _ = price("gpt-4o", input_tokens=0, output_tokens=1_000_000)

    assert cost == 10.00


def test_an_unpriced_model_reports_no_cost_rather_than_zero() -> None:
    """None says 'not known'; 0.0 would say 'free', and one of those is a lie."""
    cost, detail = price("a-model-nobody-reviewed", input_tokens=100, output_tokens=10)

    assert cost is None
    assert detail is not None
    assert "cost unknown" in detail


def test_the_in_memory_sink_keeps_records_in_order() -> None:
    sink = InMemoryTelemetry()
    sink.record(make_record(attempts=1))
    sink.record(make_record(attempts=2))

    assert [record.attempts for record in sink.records] == [1, 2]


def test_the_in_memory_sink_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryTelemetry(), TelemetrySink)


def test_unpriced_calls_make_the_total_a_lower_bound() -> None:
    sink = InMemoryTelemetry()
    sink.record(make_record(cost_usd=0.01))
    sink.record(make_record(cost_usd=None))

    assert sink.total_cost_usd == 0.01
    assert sink.unpriced_calls == 1


def test_timestamps_are_timezone_aware_utc() -> None:
    """A naive timestamp in an audit record cannot be compared across machines."""
    stamped = now()

    assert stamped.tzinfo is not None
    assert stamped.utcoffset().total_seconds() == 0
