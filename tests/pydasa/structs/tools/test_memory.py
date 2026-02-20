# -*- coding: utf-8 -*-
"""
Test Module for memory.py
===========================================

Tests for memory utility functions in PyDASA.
"""

import unittest
import unittest.mock as mock
import pytest
from dataclasses import dataclass, fields
from pydasa.structs.tools.memory import alloc_slots
from tests.pydasa.data.test_data import get_memory_test_data


class TestAllocSlots(unittest.TestCase):
    """Test cases for alloc_slots decorator."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_memory_test_data()

    def test_alloc_slots_with_regular_class(self) -> None:
        """Test alloc_slots converts regular class to dataclass with slots."""
        @alloc_slots()
        class TestClass:
            x: int
            y: str

        # Create instance and verify it works
        obj = TestClass(x=10, y="test")  # type: ignore[call-arg]
        assert obj.x == 10
        assert obj.y == "test"
        # Verify it"s a dataclass
        assert hasattr(TestClass, "__dataclass_fields__")

        # Verify slots are created
        assert hasattr(TestClass, "__slots__")

    def test_alloc_slots_with_existing_dataclass(self) -> None:
        """Test alloc_slots works with existing dataclass."""
        @alloc_slots()
        @dataclass
        class ExistingDataclass:
            value: int
            name: str

        # Verify it"s still a dataclass
        assert hasattr(ExistingDataclass, "__dataclass_fields__")

        # Verify slots are created
        assert hasattr(ExistingDataclass, "__slots__")

        # Create instance and verify it works
        obj = ExistingDataclass(value=42, name="example")  # type: ignore[call-arg]
        assert obj.value == 42
        assert obj.name == "example"

    def test_mem_slot_creates_slots(self) -> None:
        """Test that alloc_slots actually creates __slots__."""
        @alloc_slots()
        class SimpleClass:
            a: int
            b: int

        # Verify __slots__ exists and contains expected fields
        assert hasattr(SimpleClass, "__slots__")
        assert "__dict__" not in SimpleClass.__slots__  # type: ignore

    def test_alloc_slots_with_invalid_input(self) -> None:
        """Test alloc_slots raises TypeError for non-class inputs."""
        for invalid_input in self.test_data["INVALID_INPUTS"]:
            with pytest.raises(TypeError) as excinfo:
                # Apply decorator to invalid input
                alloc_slots()(invalid_input)
            assert "Invalid class" in str(excinfo.value) or "class must be a type" in str(excinfo.value)

    def test_alloc_slots_preserves_fields(self) -> None:
        """Test that alloc_slots preserves dataclass fields."""
        @alloc_slots()
        class OriginalClass:
            field1: str
            field2: int
            field3: float

        # Get field names
        field_names = {f.name for f in fields(OriginalClass)}   # type: ignore

        # Verify all original fields are preserved
        assert "field1" in field_names
        assert "field2" in field_names
        assert "field3" in field_names

    def test_alloc_slots_without_allow_dict(self) -> None:
        """Test that alloc_slots prevents dynamic attributes."""
        @alloc_slots()
        class StrictClass:
            x: int
            y: int

        obj = StrictClass(x=10, y=20)  # type: ignore[call-arg]

        # Verify slotted attributes work
        assert obj.x == 10
        assert obj.y == 20

        # Verify dynamic attributes are not allowed
        with pytest.raises(AttributeError):
            obj.z = 30  # type: ignore

    def test_alloc_slots_requires_python_310_or_higher(self) -> None:
        """Test alloc_slots raises RuntimeError on Python < 3.10."""
        # Create a mock version_info with major and minor attributes
        mock_version = mock.MagicMock()
        mock_version.__lt__ = lambda self, other: True  # Make it less than (3, 10)
        mock_version.major = 3
        mock_version.minor = 9

        # Mock sys.version_info to simulate Python 3.9
        with mock.patch('pydasa.structs.tools.memory.sys.version_info',
                        mock_version):
            with pytest.raises(RuntimeError) as exc_info:
                @alloc_slots()
                class TestClass:
                    x: int

            assert "alloc_slots requires Python 3.10+" in str(exc_info.value)
            assert "Current version: 3.9" in str(exc_info.value)
