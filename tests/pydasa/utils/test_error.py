# -*- coding: utf-8 -*-
"""
Test Module for error.py
===========================================

Tests for error handling and exceptions in PyDASA.
"""

import pytest
from pydasa.utils.error import handle_error, inspect_var


class TestHandleError:
    """Test handle_error() function.
    """

    def test_handle_error_with_valid_inputs(self) -> None:
        """Test handle_error with valid context, function, and exception.
        """
        ctx = "TestClass"
        func = "test_method"
        exc = ValueError("Test error message")

        with pytest.raises(ValueError) as exc_info:
            handle_error(ctx, func, exc)

        # Check that the error message contains context information
        assert "Error in TestClass.test_method" in str(exc_info.value)
        assert "Test error message" in str(exc_info.value)

    def test_handle_error_preserves_exception_type(self) -> None:
        """Test that handle_error preserves the original exception type.
        """
        ctx = "MyModule"
        func = "my_function"

        # Test with ValueError
        with pytest.raises(ValueError):
            handle_error(ctx, func, ValueError("test"))

        # Test with TypeError
        with pytest.raises(TypeError):
            handle_error(ctx, func, TypeError("test"))

        # Test with RuntimeError
        with pytest.raises(RuntimeError):
            handle_error(ctx, func, RuntimeError("test"))

    def test_handle_error_with_different_exception_types(self) -> None:
        """Test handle_error with various exception types.
        """
        ctx = "TestContext"
        func = "test_func"

        exceptions = [
            ValueError("value error"),
            TypeError("type error"),
            RuntimeError("runtime error"),
            KeyError("key error"),
            IndexError("index error"),
            AttributeError("attribute error"),
        ]

        for exc in exceptions:
            with pytest.raises(type(exc)) as exc_info:
                handle_error(ctx, func, exc)

            assert "Error in TestContext.test_func" in str(exc_info.value)

    def test_handle_error_invalid_context_type(self) -> None:
        """Test handle_error raises TypeError for invalid context type.
        """
        with pytest.raises(TypeError) as exc_info:
            handle_error(123, "func", ValueError("test"))

        assert "Invalid context" in str(exc_info.value)
        assert "Context must be a string" in str(exc_info.value)

    def test_handle_error_invalid_function_type(self) -> None:
        """Test handle_error raises TypeError for invalid function name type.
        """
        with pytest.raises(TypeError) as exc_info:
            handle_error("context", 456, ValueError("test"))

        assert "Invalid function name" in str(exc_info.value)
        assert "Function name must be a string" in str(exc_info.value)

    def test_handle_error_invalid_exception_type(self) -> None:
        """Test handle_error raises TypeError for invalid exception type.
        """
        with pytest.raises(TypeError) as exc_info:
            handle_error("context", "func", "not an exception")

        assert "Invalid exception" in str(exc_info.value)
        assert "Exception must be an instance of Exception" in str(exc_info.value)

    def test_handle_error_with_empty_strings(self) -> None:
        """Test handle_error with empty context and function names.
        """
        # Empty strings are valid, should not raise TypeError
        with pytest.raises(ValueError) as exc_info:
            handle_error("", "", ValueError("test"))

        assert "Error in ." in str(exc_info.value)

    def test_handle_error_preserves_traceback(self) -> None:
        """Test that handle_error preserves the original traceback.
        """
        ctx = "TestModule"
        func = "test_function"

        try:
            # Create an exception with a traceback
            raise ValueError("Original error")
        except ValueError as e:
            with pytest.raises(ValueError) as exc_info:
                handle_error(ctx, func, e)

            # Check that traceback is preserved
            assert exc_info.tb is not None

    def test_handle_error_with_nested_exceptions(self) -> None:
        """Test handle_error with nested exception messages.
        """
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

    def test_handle_error_with_unicode_context(self) -> None:
        """Test handle_error with unicode characters in context.
        """
        ctx = "TestContext_üñíçödé"
        func = "test_func_αβγ"
        exc = ValueError("Error with émojis 🚀")

        with pytest.raises(ValueError) as exc_info:
            handle_error(ctx, func, exc)

        assert ctx in str(exc_info.value)
        assert func in str(exc_info.value)

    def test_handle_error_with_long_messages(self) -> None:
        """Test handle_error with long error messages.
        """
        ctx = "LongContext"
        func = "long_function"
        long_message = "x" * 1000  # Very long error message
        exc = ValueError(long_message)

        with pytest.raises(ValueError) as exc_info:
            handle_error(ctx, func, exc)

        # Should contain the full message
        assert long_message in str(exc_info.value)


class TestInspectVar:
    """Test inspect_var() function.
    """

    def test_inspect_var_simple_variable(self) -> None:
        """Test inspect_var with a simple variable.
        """
        test_variable = 42
        result = inspect_var(test_variable)
        assert result == "test_variable"

    def test_inspect_var_string_variable(self) -> None:
        """Test inspect_var with a string variable.
        """
        my_string = "hello"
        result = inspect_var(my_string)
        assert result == "my_string"

    def test_inspect_var_list_variable(self) -> None:
        """Test inspect_var with a list variable.
        """
        my_list = [1, 2, 3]
        result = inspect_var(my_list)
        assert result == "my_list"

    def test_inspect_var_dict_variable(self) -> None:
        """Test inspect_var with a dictionary variable.
        """
        my_dict = {"key": "value"}
        result = inspect_var(my_dict)
        assert result == "my_dict"

    def test_inspect_var_object_variable(self) -> None:
        """Test inspect_var with an object variable.
        """
        class TestClass:
            pass

        my_object = TestClass()
        result = inspect_var(my_object)
        assert result == "my_object"

    def test_inspect_var_multiple_variables_same_value(self) -> None:
        """Test inspect_var when multiple variables have the same value.
        """
        value = 100
        var1 = value
        var2 = value
        value = var1 + var2

        # Should return one of the variable names
        result = inspect_var(value)
        assert result in ["value", "var1", "var2"]

    def test_inspect_var_none_variable(self) -> None:
        """Test inspect_var with None.
        """
        none_var = None
        result = inspect_var(none_var)
        assert result == "none_var"

    def test_inspect_var_boolean_variable(self) -> None:
        """Test inspect_var with boolean variables.
        """
        true_var = True
        result = inspect_var(true_var)
        assert result == "true_var"

        false_var = False
        result = inspect_var(false_var)
        assert result == "false_var"

    def test_inspect_var_numeric_types(self) -> None:
        """Test inspect_var with different numeric types.
        """
        int_var = 42
        assert inspect_var(int_var) == "int_var"

        float_var = 3.14
        assert inspect_var(float_var) == "float_var"

        complex_var = 1 + 2j
        assert inspect_var(complex_var) == "complex_var"

    def test_inspect_var_with_underscore_names(self) -> None:
        """Test inspect_var with variables containing underscores.
        """
        my_test_variable = "test"
        result = inspect_var(my_test_variable)
        assert result == "my_test_variable"

        _private_var = "private"
        result = inspect_var(_private_var)
        assert result == "_private_var"

    def test_inspect_var_tuple_variable(self) -> None:
        """Test inspect_var with a tuple variable.
        """
        my_tuple = (1, 2, 3)
        result = inspect_var(my_tuple)
        assert result == "my_tuple"

    def test_inspect_var_set_variable(self) -> None:
        """Test inspect_var with a set variable.
        """
        my_set = {1, 2, 3}
        result = inspect_var(my_set)
        assert result == "my_set"

    def test_inspect_var_lambda_variable(self) -> None:
        """Test inspect_var with a lambda function.
        """
        my_lambda = lambda x: x * 2
        result = inspect_var(my_lambda)
        assert result == "my_lambda"

    def test_inspect_var_function_variable(self) -> None:
        """Test inspect_var with a function variable.
        """
        def my_function():
            pass

        result = inspect_var(my_function)
        assert result == "my_function"

    def test_inspect_var_mutable_vs_immutable(self) -> None:
        """Test inspect_var with mutable and immutable objects.
        """
        # Mutable object
        mutable_list = [1, 2, 3]
        result = inspect_var(mutable_list)
        assert result == "mutable_list"

        # Immutable object
        immutable_tuple = (1, 2, 3)
        result = inspect_var(immutable_tuple)
        assert result == "immutable_tuple"


class TestErrorHandlingIntegration:
    """Integration tests for error handling.
    """

    def test_handle_error_in_real_scenario(self) -> None:
        """Test handle_error in a realistic error handling scenario.
        """
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
        """Test using inspect_var before handle_error.
        """
        test_var = "test_value"
        var_name = inspect_var(test_var)

        assert var_name == "test_var"

        # Now use this in an error context
        exc = ValueError(f"Invalid value for {var_name}")
        with pytest.raises(ValueError) as exc_info:
            handle_error("TestModule", "test_function", exc)

        assert "Invalid value for test_var" in str(exc_info.value)

    def test_error_handling_chain(self) -> None:
        """Test chaining multiple error handlers.
        """
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

        # Should have the outer context
        assert "Error in OuterModule.outer_function" in str(exc_info.value)

    def test_custom_exception_with_handle_error(self) -> None:
        """Test handle_error with custom exception classes.
        """
        class CustomError(Exception):
            pass

        ctx = "CustomModule"
        func = "custom_function"
        exc = CustomError("Custom error occurred")

        with pytest.raises(CustomError) as exc_info:
            handle_error(ctx, func, exc)

        assert "Error in CustomModule.custom_function" in str(exc_info.value)
        assert "Custom error occurred" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and boundary conditions.
        """
    
    def test_handle_error_with_special_characters(self) -> None:
        """Test handle_error with special characters in messages.
        """
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        ctx = "TestContext"
        func = "test_func"
        exc = ValueError(special_chars)

        with pytest.raises(ValueError) as exc_info:
            handle_error(ctx, func, exc)

        assert special_chars in str(exc_info.value)

    def test_inspect_var_with_very_long_name(self) -> None:
        """Test inspect_var with very long variable names.
        """
        very_long_variable_name_that_exceeds_normal_length = 42
        result = inspect_var(very_long_variable_name_that_exceeds_normal_length)
        assert result == "very_long_variable_name_that_exceeds_normal_length"

    def test_handle_error_with_multiline_message(self) -> None:
        """Test handle_error with multiline error messages.
        """
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

    def test_inspect_var_same_id_different_variables(self) -> None:
        """Test inspect_var with variables that might have same id.
        """
        # Small integers are cached in Python
        x = 5
        y = 5

        # Should return one of them
        result = inspect_var(x)
        result = inspect_var(y)
        assert result in ["x", "y"]
