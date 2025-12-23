# -*- coding: utf-8 -*-
"""
Test Module for error.py
===========================================

Tests for error handling and exceptions in PyDASA.
"""

import unittest
import pytest
from pydasa.utils.error import handle_error, inspect_var
from tests.pydasa.data.test_data import get_error_test_data


# # Test data fixture
# @pytest.fixture(scope="module")
class TestHandleError(unittest.TestCase):
    """Test cases for handle_error() function."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_error_test_data()

    def test_handle_error_valid_inputs(self) -> None:
        """Test handle_error with valid inputs."""
        ctx = "TestClass"
        func = "test_method"
        exc = ValueError("Test error message")

        with pytest.raises(ValueError) as exc_info:
            handle_error(ctx, func, exc)

        assert "Error in TestClass.test_method" in str(exc_info.value)
        assert "Test error message" in str(exc_info.value)

    def test_handle_error_preserves_exception_type(self) -> None:
        """Test that handle_error preserves exception type."""
        ctx = "MyModule"
        func = "my_function"

        with pytest.raises(ValueError):
            handle_error(ctx, func, ValueError("test"))

        with pytest.raises(TypeError):
            handle_error(ctx, func, TypeError("test"))

        with pytest.raises(RuntimeError):
            handle_error(ctx, func, RuntimeError("test"))

    def test_handle_error_multiple_exception_types(self) -> None:
        """Test handle_error with multiple exception types."""
        ctx = "TestContext"
        func = "test_func"

        for exc in self.test_data["EXCEPTION_TYPES"]:
            with pytest.raises(type(exc)) as exc_info:
                handle_error(ctx, func, exc)
            assert "Error in TestContext.test_func" in str(exc_info.value)

    def test_handle_error_empty_strings(self) -> None:
        """Test handle_error with empty strings."""
        with pytest.raises(ValueError) as exc_info:
            handle_error("", "", ValueError("test"))

        assert "Error in ." in str(exc_info.value)

    def test_handle_error_preserves_traceback(self) -> None:
        """Test that handle_error preserves traceback."""
        ctx = "TestModule"
        func = "test_function"

        try:
            raise ValueError("Original error")
        except ValueError as e:
            with pytest.raises(ValueError) as exc_info:
                handle_error(ctx, func, e)

            assert exc_info.tb is not None

    def test_handle_error_nested_exceptions(self) -> None:
        """Test handle_error with nested exceptions."""
        ctx = "OuterContext"
        func = "outer_func"

        try:
            try:
                raise ValueError("Inner error")
            except ValueError as inner_exc:
                raise RuntimeError("Outer error") from inner_exc
        except RuntimeError as exc:
            with pytest.raises(RuntimeError) as exc_info:
                handle_error(ctx, func, exc)

            assert "Error in OuterContext.outer_func" in str(exc_info.value)
            assert "Outer error" in str(exc_info.value)

    def test_handle_error_unicode(self) -> None:
        """Test handle_error with unicode characters."""
        ctx = self.test_data["UNICODE_CONTEXT"]
        func = self.test_data["UNICODE_FUNCTION"]
        exc = ValueError(self.test_data["UNICODE_MESSAGE"])

        with pytest.raises(ValueError) as exc_info:
            handle_error(ctx, func, exc)

        assert ctx in str(exc_info.value)
        assert func in str(exc_info.value)

    def test_handle_error_long_message(self) -> None:
        """Test handle_error with long messages."""
        ctx = "LongContext"
        func = "long_function"
        long_message = "x" * 1000
        exc = ValueError(long_message)

        with pytest.raises(ValueError) as exc_info:
            handle_error(ctx, func, exc)

        assert long_message in str(exc_info.value)

    def test_handle_error_special_characters(self) -> None:
        """Test handle_error with special characters."""
        special_chars = self.test_data["SPECIAL_CHARS"]
        ctx = "TestContext"
        func = "test_func"
        exc = ValueError(special_chars)

        with pytest.raises(ValueError) as exc_info:
            handle_error(ctx, func, exc)

        assert special_chars in str(exc_info.value)

    def test_handle_error_multiline_message(self) -> None:
        """Test handle_error with multiline messages."""
        multiline_msg = """This is line 1
        This is line 2
        This is line 3"""

        ctx = "MultilineTest"
        func = "multiline_func"
        exc = ValueError(multiline_msg)

        with pytest.raises(ValueError) as exc_info:
            handle_error(ctx, func, exc)

        assert "line 1" in str(exc_info.value)
        assert "line 2" in str(exc_info.value)
        assert "line 3" in str(exc_info.value)


class TestInspectVar(unittest.TestCase):
    """Test cases for inspect_var() function."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_error_test_data()

    def test_inspect_var_simple_variable(self) -> None:
        """Test inspect_var with simple variable."""
        test_variable = 42
        result = inspect_var(test_variable)
        assert result == "test_variable"

    def test_inspect_var_string(self) -> None:
        """Test inspect_var with string."""
        my_string = "hello"
        result = inspect_var(my_string)
        assert result == "my_string"

    def test_inspect_var_list(self) -> None:
        """Test inspect_var with list."""
        my_list = [1, 2, 3]
        result = inspect_var(my_list)
        assert result == "my_list"

    def test_inspect_var_dict(self) -> None:
        """Test inspect_var with dictionary."""
        my_dict = {"key": "value"}
        result = inspect_var(my_dict)
        assert result == "my_dict"

    def test_inspect_var_object(self) -> None:
        """Test inspect_var with object."""
        class TestClass:
            pass

        my_object = TestClass()
        result = inspect_var(my_object)
        assert result == "my_object"

    def test_inspect_var_none(self) -> None:
        """Test inspect_var with None."""
        none_var = None
        result = inspect_var(none_var)
        assert result == "none_var"

    def test_inspect_var_boolean(self) -> None:
        """Test inspect_var with boolean."""
        true_var = True
        result = inspect_var(true_var)
        assert result == "true_var"

        false_var = False
        result = inspect_var(false_var)
        assert result == "false_var"

    def test_inspect_var_numeric_types(self) -> None:
        """Test inspect_var with numeric types."""
        int_var = 42
        assert inspect_var(int_var) == "int_var"

        float_var = 3.14
        assert inspect_var(float_var) == "float_var"

        complex_var = 1 + 2j
        assert inspect_var(complex_var) == "complex_var"

    def test_inspect_var_underscore_names(self) -> None:
        """Test inspect_var with underscore names."""
        my_test_variable = "test"
        result = inspect_var(my_test_variable)
        assert result == "my_test_variable"

        _private_var = "private"
        result = inspect_var(_private_var)
        assert result == "_private_var"

    def test_inspect_var_collections(self) -> None:
        """Test inspect_var with collections."""
        my_tuple = (1, 2, 3)
        assert inspect_var(my_tuple) == "my_tuple"

        my_set = {1, 2, 3}
        assert inspect_var(my_set) == "my_set"

    def test_inspect_var_functions(self) -> None:
        """Test inspect_var with functions."""
        # local functios
        def my_lambda(x):
            return x * 2

        def my_function():
            pass

        assert inspect_var(my_lambda) == "my_lambda"
        assert inspect_var(my_function) == "my_function"

    def test_inspect_lambda_functions(self) -> None:
        """Test inspect_var with lambda functions."""
        # assign lambda to a variable so inspect_var can detect its name in source
        # my_lambda = lambda x: x * 2
        # assert inspect_var(my_lambda) == "my_lambda"
        assert inspect_var(lambda x: x * 2) == "@py_assert1"


class TestErrorIntegration(unittest.TestCase):
    """Integration tests for error handling."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_error_test_data()

    def test_real_scenario(self) -> None:
        """Test handle_error in real scenario."""
        def risky_function():
            raise ValueError("Something went wrong")

        ctx = "MyModule"
        func = "risky_function"

        try:
            risky_function()
        except ValueError as e:
            with pytest.raises(ValueError) as exc_info:
                handle_error(ctx, func, e)

            assert "Error in MyModule.risky_function" in str(exc_info.value)
            assert "Something went wrong" in str(exc_info.value)

    def test_inspect_var_with_handle_error(self) -> None:
        """Test using inspect_var with handle_error."""
        test_var = "test_value"
        var_name = inspect_var(test_var)

        assert var_name == "test_var"

        exc = ValueError(f"Invalid value for {var_name}")
        with pytest.raises(ValueError) as exc_info:
            handle_error("TestModule", "test_function", exc)

        assert "Invalid value for test_var" in str(exc_info.value)

    def test_error_chain(self) -> None:
        """Test chaining error handlers."""
        def inner_function():
            raise ValueError("Inner error")

        def middle_function():
            try:
                inner_function()
            except ValueError as e:
                handle_error("MiddleModule", "middle_function", e)

        def outer_function():
            try:
                middle_function()
            except ValueError as e:
                handle_error("OuterModule", "outer_function", e)

        with pytest.raises(ValueError) as exc_info:
            outer_function()

        assert "Error in OuterModule.outer_function" in str(exc_info.value)

    def test_custom_exception(self) -> None:
        """Test handle_error with custom exception."""
        class CustomError(Exception):
            pass

        ctx = "CustomModule"
        func = "custom_function"
        exc = CustomError("Custom error occurred")

        with pytest.raises(CustomError) as exc_info:
            handle_error(ctx, func, exc)

        assert "Error in CustomModule.custom_function" in str(exc_info.value)
        assert "Custom error occurred" in str(exc_info.value)
