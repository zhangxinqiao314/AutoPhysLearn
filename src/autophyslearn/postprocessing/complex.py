import torch
class ComplexPostProcessor:
    """
    A class used to post-process complex numbers from model fits by scaling their real and imaginary components 
    based on dataset scalers. The results are stacked along a new dimension.

    Attributes:
        dataset (object): The dataset containing the scalers used for normalizing the real and imaginary parts.
    """

    def __init__(self, dataset):
        """
        Initializes the ComplexPostProcessor with the given dataset.

        Args:
            dataset (object): The dataset object containing scalers for real and imaginary components.
        """
        self.dataset = dataset

    def compute(self, fits):
        """
        Processes the complex fits by extracting, scaling, and stacking the real and imaginary components.

        Args:
            fits (torch.Tensor): A tensor of complex numbers from which real and imaginary components are extracted.

        Returns:
            torch.Tensor: A tensor with the scaled real and imaginary components stacked along the last dimension.
        """

        # Extract the real part of the complex tensor
        real = torch.real(fits)

        # Scale the real part using the dataset's real scaler, moving data to the GPU (cuda)
        real_scaled = (real - torch.tensor(self.dataset.raw_data_scaler.real_scaler.mean).cuda()) / \
                      torch.tensor(self.dataset.raw_data_scaler.real_scaler.std).cuda()

        # Extract the imaginary part of the complex tensor
        imag = torch.imag(fits)

        # Scale the imaginary part using the dataset's imaginary scaler, moving data to the GPU (cuda)
        imag_scaled = (imag - torch.tensor(self.dataset.raw_data_scaler.imag_scaler.mean).cuda()) / \
                      torch.tensor(self.dataset.raw_data_scaler.imag_scaler.std).cuda()

        # Stack the scaled real and imaginary components along a new dimension
        out = torch.stack((real_scaled, imag_scaled), dim=2)

        return out
