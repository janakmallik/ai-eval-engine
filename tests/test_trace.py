from datetime import datetime, timezone

from aieval.tracing.trace import Trace


def test_trace_has_id():
    trace = Trace()

    assert trace.trace_id
    assert isinstance(trace.trace_id, str)


def test_trace_has_start_timestamp():
    trace = Trace()

    assert trace.started_at is not None
    assert trace.started_at.tzinfo == timezone.utc


def test_trace_is_not_ended_when_created():
    trace = Trace()

    assert trace.ended_at is None


def test_trace_can_be_ended():
    trace = Trace()

    trace.end()

    assert trace.ended_at is not None
    assert trace.ended_at.tzinfo == timezone.utc


def test_trace_duration_is_none_before_end():
    trace = Trace()

    assert trace.duration is None


def test_trace_duration_is_non_negative_after_end():
    trace = Trace()

    trace.end()

    assert trace.duration is not None
    assert trace.duration >= 0


def test_trace_ids_are_unique():
    first = Trace()
    second = Trace()

    assert first.trace_id != second.trace_id