# -*- coding: utf-8 -*-
"""
Test Module io.py
===========================================

Tests for input/output functions in PyDASA.utils.io module.
"""
import pytest
import json
import os
import tempfile
from pydasa.utils import io


class TestLoad:
    """Test suite for the load() function."""

    def test_load_valid_json(self):
        """Test loading a valid JSON file."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
            test_data = {'key': 'value', 'number': 42, 'list': [1, 2, 3]}
            json.dump(test_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            # Load the data
            loaded_data = io.load(tmp_path)
            assert loaded_data == test_data
            assert loaded_data['key'] == 'value'
            assert loaded_data['number'] == 42
            assert loaded_data['list'] == [1, 2, 3]
        finally:
            # Clean up
            os.unlink(tmp_path)

    def test_load_empty_json(self):
        """Test loading an empty JSON object."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
            json.dump({}, tmp_file)
            tmp_path = tmp_file.name

        try:
            loaded_data = io.load(tmp_path)
            assert loaded_data == {}
        finally:
            os.unlink(tmp_path)

    def test_load_nested_json(self):
        """Test loading a nested JSON structure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
            test_data = {
                'nested': {
                    'level1': {
                        'level2': 'value'
                    }
                },
                'array': [{'id': 1}, {'id': 2}]
            }
            json.dump(test_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            loaded_data = io.load(tmp_path)
            assert loaded_data == test_data
            assert loaded_data['nested']['level1']['level2'] == 'value'
            assert loaded_data['array'][0]['id'] == 1
        finally:
            os.unlink(tmp_path)

    def test_load_file_not_found(self):
        """Test loading a file that does not exist."""
        with pytest.raises(FileNotFoundError):
            io.load('/nonexistent/path/to/file.json')

    def test_load_invalid_json(self):
        """Test loading a file with invalid JSON content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write('{ invalid json content }')
            tmp_path = tmp_file.name

        try:
            with pytest.raises(json.JSONDecodeError):
                io.load(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_load_unicode_content(self):
        """Test loading JSON with Unicode characters."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
            test_data = {'text': 'Hello 世界', 'emoji': '🚀'}
            json.dump(test_data, tmp_file, ensure_ascii=False)
            tmp_path = tmp_file.name

        try:
            loaded_data = io.load(tmp_path)
            assert loaded_data['text'] == 'Hello 世界'
            assert loaded_data['emoji'] == '🚀'
        finally:
            os.unlink(tmp_path)

    def test_load_with_bom(self):
        """Test loading JSON with UTF-8 BOM (Byte Order Mark)."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False) as tmp_file:
            test_data = {'key': 'value', 'number': 42}
            # Write BOM followed by JSON data
            tmp_file.write(b'\xef\xbb\xbf')  # UTF-8 BOM
            tmp_file.write(json.dumps(test_data).encode('utf-8'))
            tmp_path = tmp_file.name

        try:
            loaded_data = io.load(tmp_path)
            assert loaded_data == test_data
            assert loaded_data['key'] == 'value'
            assert loaded_data['number'] == 42
        finally:
            os.unlink(tmp_path)


class TestSave:
    """Test suite for the save() function."""

    def test_save_valid_data(self):
        """Test saving valid data to a JSON file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'test.json')
            test_data = {'key': 'value', 'number': 42, 'list': [1, 2, 3]}

            # Save the data
            io.save(test_data, tmp_path)

            # Verify the file was created and contains correct data
            assert os.path.exists(tmp_path)
            with open(tmp_path, 'r', encoding='utf-8') as file:
                loaded_data = json.load(file)
                assert loaded_data == test_data

    def test_save_empty_dict(self):
        """Test saving an empty dictionary."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'empty.json')
            io.save({}, tmp_path)

            assert os.path.exists(tmp_path)
            with open(tmp_path, 'r', encoding='utf-8') as file:
                loaded_data = json.load(file)
                assert loaded_data == {}

    def test_save_nested_data(self):
        """Test saving nested data structures."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'nested.json')
            test_data = {
                'nested': {
                    'level1': {
                        'level2': 'value'
                    }
                },
                'array': [{'id': 1}, {'id': 2}]
            }
            io.save(test_data, tmp_path)

            with open(tmp_path, 'r', encoding='utf-8') as file:
                loaded_data = json.load(file)
                assert loaded_data == test_data

    def test_save_creates_directories(self):
        """Test that save creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_path = os.path.join(tmp_dir, 'subdir1', 'subdir2', 'test.json')
            test_data = {'key': 'value'}

            io.save(test_data, nested_path)

            assert os.path.exists(nested_path)
            with open(nested_path, 'r', encoding='utf-8') as file:
                loaded_data = json.load(file)
                assert loaded_data == test_data

    def test_save_overwrites_existing_file(self):
        """Test that save overwrites existing files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'test.json')

            # Save initial data
            initial_data = {'key': 'initial'}
            io.save(initial_data, tmp_path)

            # Save new data
            new_data = {'key': 'updated'}
            io.save(new_data, tmp_path)

            # Verify new data is saved
            with open(tmp_path, 'r', encoding='utf-8') as file:
                loaded_data = json.load(file)
                assert loaded_data == new_data

    def test_save_unicode_content(self):
        """Test saving data with Unicode characters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'unicode.json')
            test_data = {'text': 'Hello 世界', 'emoji': '🚀'}

            io.save(test_data, tmp_path)

            with open(tmp_path, 'r', encoding='utf-8') as file:
                loaded_data = json.load(file)
                assert loaded_data['text'] == 'Hello 世界'
                assert loaded_data['emoji'] == '🚀'

    def test_save_formatting(self):
        """Test that saved JSON is properly formatted with indentation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'formatted.json')
            test_data = {'key1': 'value1', 'key2': 'value2'}

            io.save(test_data, tmp_path)

            # Read the file as text to check formatting
            with open(tmp_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Check that the JSON is indented (not all on one line)
                assert '\n' in content
                assert '    ' in content  # Check for indentation


class TestRoundTrip:
    """Test suite for save and load round-trip operations."""

    def test_round_trip_simple(self):
        """Test that data saved and loaded matches the original."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'roundtrip.json')
            original_data = {
                'string': 'test',
                'integer': 123,
                'float': 45.67,
                'boolean': True,
                'null': None,
                'list': [1, 2, 3],
                'dict': {'nested': 'value'}
            }

            io.save(original_data, tmp_path)
            loaded_data = io.load(tmp_path)

            assert loaded_data == original_data

    def test_round_trip_complex(self):
        """Test round-trip with complex nested structures."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'complex.json')
            original_data = {
                'metadata': {
                    'version': '1.0',
                    'author': 'Test'
                },
                'data': [
                    {'id': 1, 'values': [1.1, 2.2, 3.3]},
                    {'id': 2, 'values': [4.4, 5.5, 6.6]}
                ],
                'config': {
                    'enabled': True,
                    'options': None
                }
            }

            io.save(original_data, tmp_path)
            loaded_data = io.load(tmp_path)

            assert loaded_data == original_data
