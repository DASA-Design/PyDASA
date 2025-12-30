# -*- coding: utf-8 -*-
"""
Test Module for io.py
===========================================

Tests for generic I/O operations in PyDASA.
"""

import unittest
import pytest
import json
import tempfile
from pathlib import Path
from pydasa.utils.io import load_json, save_json, load, save
from tests.pydasa.data.test_data import get_io_test_data


class TestIO(unittest.TestCase):
    """Test cases for I/O module."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_io_test_data()

    def setUp(self) -> None:
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    # load_json() tests
    def test_load_json_simple(self) -> None:
        """Test loading simple JSON file."""
        test_file = self.temp_path / self.test_data["TEST_FILENAMES"]["simple"]

        with open(test_file, "w") as f:
            json.dump(self.test_data["SIMPLE_JSON"], f)

        result = load_json(test_file)
        assert result == self.test_data["SIMPLE_JSON"]

    def test_load_json_nested(self) -> None:
        """Test loading nested JSON structure."""
        test_file = self.temp_path / self.test_data["TEST_FILENAMES"]["nested"]

        with open(test_file, "w") as f:
            json.dump(self.test_data["NESTED_JSON"], f)

        result = load_json(test_file)
        assert result["level1"]["level2"]["value"] == "nested"

    def test_load_json_unicode(self) -> None:
        """Test loading JSON with unicode characters."""
        test_file = self.temp_path / self.test_data["TEST_FILENAMES"]["unicode"]

        with open(test_file, "w", encoding='utf-8') as f:
            json.dump(self.test_data["UNICODE_JSON"], f, ensure_ascii=False)

        result = load_json(test_file)
        assert result["text"] == self.test_data["UNICODE_JSON"]["text"]
        assert result["emoji"] == self.test_data["UNICODE_JSON"]["emoji"]

    def test_load_json_file_not_found(self) -> None:
        """Test error when file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            load_json(self.temp_path / "nonexistent.json")

    def test_load_json_invalid_json(self) -> None:
        """Test error with invalid JSON."""
        test_file = self.temp_path / self.test_data["TEST_FILENAMES"]["invalid"]
        with open(test_file, "w") as f:
            f.write(self.test_data["INVALID_JSON_CONTENT"])

        with self.assertRaises(json.JSONDecodeError):
            load_json(test_file)

    # save_json() tests
    def test_save_json_simple(self) -> None:
        """Test saving simple JSON file."""
        test_file = self.temp_path / self.test_data["TEST_FILENAMES"]["output"]

        save_json(self.test_data["SIMPLE_JSON"], test_file)

        assert test_file.exists()
        with open(test_file, "r") as f:
            result = json.load(f)
        assert result == self.test_data["SIMPLE_JSON"]

    def test_save_json_creates_directory(self) -> None:
        """Test that save_json creates parent directories."""
        test_file = self.temp_path / self.test_data["TEST_FILENAMES"]["nested_dir"]

        save_json({"created": True}, test_file)

        assert test_file.exists()
        assert test_file.parent.exists()

    def test_save_json_custom_indent(self) -> None:
        """Test saving with custom indentation."""
        test_file = self.temp_path / "indent.json"

        save_json(self.test_data["SIMPLE_JSON"], test_file, indent=2)

        with open(test_file, "r") as f:
            content = f.read()
        assert "  " in content

    def test_save_json_unicode(self) -> None:
        """Test saving JSON with unicode characters."""
        test_file = self.temp_path / self.test_data["TEST_FILENAMES"]["unicode"]

        save_json(self.test_data["UNICODE_JSON"], test_file)

        with open(test_file, "r", encoding='utf-8') as f:
            result = json.load(f)
        assert result["text"] == self.test_data["UNICODE_JSON"]["text"]

    # load() generic function tests
    def test_load_detects_json(self) -> None:
        """Test that load() correctly handles .json extension."""
        test_file = self.temp_path / "data.json"
        test_data = {"format": "json"}

        with open(test_file, "w") as f:
            json.dump(test_data, f)

        result = load(test_file)
        assert result == test_data

    def test_load_unsupported_formats(self) -> None:
        """Test error with unsupported file formats."""
        for ext in self.test_data["UNSUPPORTED_FORMATS"]:
            test_file = self.temp_path / f"data{ext}"

            with self.assertRaises(ValueError) as context:
                load(test_file)
            assert "Unsupported file format" in str(context.exception)

    def test_load_accepts_string_path(self) -> None:
        """Test that load() accepts string paths."""
        test_file = self.temp_path / "string.json"

        with open(test_file, "w") as f:
            json.dump(self.test_data["SIMPLE_JSON"], f)

        result = load(str(test_file))
        assert result == self.test_data["SIMPLE_JSON"]

    # save() generic function tests
    def test_save_detects_json(self) -> None:
        """Test that save() correctly handles .json extension."""
        test_file = self.temp_path / self.test_data["TEST_FILENAMES"]["output"]

        save(self.test_data["SIMPLE_JSON"], test_file)

        with open(test_file, "r") as f:
            result = json.load(f)
        assert result == self.test_data["SIMPLE_JSON"]

    def test_save_with_kwargs(self) -> None:
        """Test that save() passes kwargs to format handler."""
        test_file = self.temp_path / "indent.json"

        save(self.test_data["SIMPLE_JSON"], test_file, indent=2)

        with open(test_file, "r") as f:
            content = f.read()
        assert "  " in content

    def test_save_unsupported_formats(self) -> None:
        """Test error with unsupported file formats."""
        for ext in self.test_data["UNSUPPORTED_FORMATS"]:
            test_file = self.temp_path / f"data{ext}"

            with self.assertRaises(ValueError) as context:
                save({"data": "test"}, test_file)
            assert "Unsupported file format" in str(context.exception)

    def test_save_accepts_string_path(self) -> None:
        """Test that save() accepts string paths."""
        test_file = self.temp_path / "string.json"

        save(self.test_data["SIMPLE_JSON"], str(test_file))

        assert test_file.exists()

    def test_save_load_roundtrip(self) -> None:
        """Test that save and load operations are consistent."""
        test_file = self.temp_path / "roundtrip.json"

        save(self.test_data["NESTED_JSON"], test_file)
        result = load(test_file)

        assert result == self.test_data["NESTED_JSON"]

    def test_array_data_handling(self) -> None:
        """Test handling of JSON with arrays."""
        test_file = self.temp_path / "array.json"

        save_json(self.test_data["ARRAY_JSON"], test_file)
        result = load_json(test_file)

        assert result["items"] == self.test_data["ARRAY_JSON"]["items"]
        assert result["names"] == self.test_data["ARRAY_JSON"]["names"]
