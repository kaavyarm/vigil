from process_one_patient import time_to_minutes, clean_value


def test_time_to_minutes_zero():
    assert time_to_minutes("00:00") == 0


def test_time_to_minutes_one_hour():
    assert time_to_minutes("01:00") == 60


def test_time_to_minutes_half_hour():
    assert time_to_minutes("00:30") == 30


def test_time_to_minutes_complex():
    assert time_to_minutes("12:30") == 750


def test_clean_value_negative_one_returns_none():
    assert clean_value(-1) is None


def test_clean_value_float_returns_float():
    assert clean_value(80.5) == 80.5


def test_clean_value_integer_float_returns_int():
    result = clean_value(80.0)
    assert result == 80
    assert isinstance(result, int)


def test_clean_value_string_passthrough():
    assert clean_value("text") == "text"


def test_clean_value_zero_is_not_filtered():
    assert clean_value(0) == 0
