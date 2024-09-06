import pytest
import torch
from unittest.mock import MagicMock
from autophyslearn.postprocessing.complex import ComplexPostProcessor  # Adjust based on your project structure


@pytest.fixture
def mock_dataset():
    """Fixture to create a mock dataset with real and imaginary scalers."""
    dataset = MagicMock()
    # Mocking the scalers for real and imaginary parts
    dataset.raw_data_scaler.real_scaler.mean = 1.0
    dataset.raw_data_scaler.real_scaler.std = 0.5
    dataset.raw_data_scaler.imag_scaler.mean = 2.0
    dataset.raw_data_scaler.imag_scaler.std = 1.0
    return dataset

def test_compute_real_imag(mock_dataset):
    """Test that the compute method properly scales and stacks the real and imaginary components."""
    # Initialize the ComplexPostProcessor with the mock dataset
    processor = ComplexPostProcessor(dataset=mock_dataset)
    
    # Use torch.device to manage GPU/CPU computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a sample complex tensor
    fits = torch.tensor([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=torch.cfloat, device=device)

    # Call the compute method
    result = processor.compute(fits)

    # Expected real and imaginary components after scaling
    expected_real_scaled = (torch.real(fits) - 1.0) / 0.5  # Using mock real_scaler (mean=1.0, std=0.5)
    expected_imag_scaled = (torch.imag(fits) - 2.0) / 1.0  # Using mock imag_scaler (mean=2.0, std=1.0)

    # Stack the expected real and imaginary results along the last dimension
    expected_result = torch.stack((expected_real_scaled, expected_imag_scaled), dim=2)

    # Assert that the result matches the expected result
    assert torch.allclose(result, expected_result), "The computed result does not match the expected values."

def test_invalid_data_type(mock_dataset):
    """Test that the compute method raises an error for invalid data types."""
    processor = ComplexPostProcessor(dataset=mock_dataset)
    
    # Use torch.device to manage GPU/CPU computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create an invalid input (not complex tensor)
    fits = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float, device=device)

    # Expect an exception to be raised when processing non-complex data
    with pytest.raises(RuntimeError, match="imag is not implemented for"):
        processor.compute(fits)
