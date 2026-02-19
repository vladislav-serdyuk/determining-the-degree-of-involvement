import numpy as np

from api.stream import convert_to_serializable


class TestConvertToSerializable:
    def test_dict_with_numpy_integer(self):
        data = {"key": np.int64(42)}
        result = convert_to_serializable(data)
        assert result == {"key": 42}
        assert isinstance(result["key"], int)

    def test_dict_with_numpy_float(self):
        data = {"key": np.float64(3.14)}
        result = convert_to_serializable(data)
        assert result == {"key": 3.14}
        assert isinstance(result["key"], float)

    def test_dict_with_numpy_bool(self):
        data = {"key": np.bool_(True)}
        result = convert_to_serializable(data)
        assert result == {"key": True}
        assert isinstance(result["key"], bool)

    def test_list_with_numpy_types(self):
        data = [np.int64(1), np.float64(2.5), np.bool_(False)]
        result = convert_to_serializable(data)
        assert result == [1, 2.5, False]

    def test_tuple_with_numpy_types(self):
        data = (np.int64(1), np.float64(2.5))
        result = convert_to_serializable(data)
        assert result == [1, 2.5]

    def test_nested_dict(self):
        data = {"outer": {"inner": np.int64(42)}}
        result = convert_to_serializable(data)
        assert result == {"outer": {"inner": 42}}

    def test_numpy_array(self):
        data = {"arr": np.array([1, 2, 3])}
        result = convert_to_serializable(data)
        assert result == {"arr": [1, 2, 3]}

    def test_mixed_types(self):
        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "np_int": np.int64(10),
            "np_float": np.float64(2.5),
            "np_bool": np.bool_(True),
            "list": [1, 2, np.int64(3)],
            "nested": {"key": np.float64(1.5)}
        }
        result = convert_to_serializable(data)
        assert result == {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "np_int": 10,
            "np_float": 2.5,
            "np_bool": True,
            "list": [1, 2, 3],
            "nested": {"key": 1.5}
        }

    def test_no_numpy_types(self):
        data = {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}}
        result = convert_to_serializable(data)
        assert result == data

    def test_empty_dict(self):
        data = {}
        result = convert_to_serializable(data)
        assert result == {}

    def test_empty_list(self):
        data = []
        result = convert_to_serializable(data)
        assert result == []
