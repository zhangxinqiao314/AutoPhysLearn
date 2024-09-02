import torch
import torch.nn as nn

class Multiscale1DFitter(nn.Module):
    """
    A neural network model for fitting 1D multiscale data using a combination of 1D convolutional layers and fully connected layers.
    The model can optionally apply post-processing and scaling to the outputs.

    Attributes:
        function (callable): A function that performs the fitting process.
        x_data (torch.Tensor): Input data used for the fitting function.
        input_channels (int): Number of input channels for the convolutional layers.
        num_params (int): Number of output parameters.
        scaler (object, optional): Scaler used to unscale the output parameters.
        post_processing (object, optional): Post-processor for additional processing of the fits.
        device (str): Device to run the computations on, default is "cuda".
        loops_scaler (object, optional): Scaler used for final output scaling.
    """

    def __init__(self, function, x_data, input_channels, num_params, scaler=None, post_processing=None, device="cuda", loops_scaler=None, **kwargs):
        """
        Initializes the Multiscale1DFitter model.

        Args:
            function (callable): A function that performs the fitting process.
            x_data (torch.Tensor): Input data used for the fitting function.
            input_channels (int): Number of input channels for the convolutional layers.
            num_params (int): Number of output parameters.
            scaler (object, optional): Scaler used to unscale the output parameters.
            post_processing (object, optional): Post-processor for additional processing of the fits.
            device (str): Device to run the computations on, default is "cuda".
            loops_scaler (object, optional): Scaler used for final output scaling.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        # TODO: Could add a decoder encoder block to the model
        self.input_channels = input_channels
        self.scaler = scaler
        self.function = function
        self.x_data = x_data
        self.post_processing = post_processing
        self.device = device
        self.num_params = num_params
        self.loops_scaler = loops_scaler

        # Input block of 1D convolutional layers
        self.hidden_x1 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_channels, out_channels=8, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=8, out_channels=6, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=6, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.AdaptiveAvgPool1d(64)  # Adaptive average pooling to reduce output dimensionality
        )

        # Fully connected block
        self.hidden_xfc = nn.Sequential(
            nn.Linear(256, 64),
            nn.SELU(),
            nn.Linear(64, 32),
            nn.SELU(),
            nn.Linear(32, 20),
            nn.SELU(),
        )

        # Second block of 1D convolutional layers
        self.hidden_x2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),  # Max pooling to reduce dimensionality
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.AdaptiveAvgPool1d(16),  # Adaptive average pooling layer
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AdaptiveAvgPool1d(8),  # Adaptive average pooling layer
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AdaptiveAvgPool1d(4),  # Adaptive average pooling layer
        )

        # Flatten layer to prepare data for the fully connected layers
        self.flatten_layer = nn.Flatten()

        # Final embedding block - outputs the desired number of parameters
        self.hidden_embedding = nn.Sequential(
            nn.Linear(28, 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
            nn.Linear(8, self.num_params),
        )

    def forward(self, x, n=-1):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_channels, sequence_length).
            n (int): Batch size for reshaping. Default is -1, which means the batch size will be inferred.

        Returns:
            torch.Tensor: Scaled output fits.
            torch.Tensor: Unscaled parameters.
            torch.Tensor (optional): Embeddings, returned if not in training mode.
        """
        # Swap axes to have the correct shape for convolutional layers
        x = torch.swapaxes(x, 1, 2)
        x = self.hidden_x1(x)
        
        # Reshape the output for the fully connected block
        xfc = torch.reshape(x, (n, 256))  # (batch_size, features)
        xfc = self.hidden_xfc(xfc)

        # Reshape for the second block of convolutional layers
        x = torch.reshape(x, (n, 2, 128))
        x = self.hidden_x2(x)
        
        # Flatten the output of the second convolutional block
        cnn_flat = self.flatten_layer(x)

        # Combine the flattened convolutional output and fully connected output
        encoded = torch.cat((cnn_flat, xfc), dim=1)
        
        # Get the final embedding (output parameters)
        embedding = self.hidden_embedding(encoded)

        unscaled_param = embedding

        # If a scaler is provided, unscale the parameters
        if self.scaler is not None:
            unscaled_param = (
                embedding * torch.tensor(self.scaler.var_ ** 0.5).cuda() 
                + torch.tensor(self.scaler.mean_).cuda()
            )

        # Pass the unscaled parameters to the fitting function
        fits = self.function(unscaled_param, self.x_data, device=self.device)

        out = fits

        # If post-processing is required, apply it
        if self.post_processing is not None:
            out = self.post_processing.compute(fits)
        else:
            out = fits

        # If a loops scaler is provided, scale the final output
        if self.loops_scaler is not None:
            out_scaled = (out - torch.tensor(self.loops_scaler.mean).cuda()) / \
                          torch.tensor(self.loops_scaler.std).cuda()
        else:
            out_scaled = out

        if self.training:
            return out_scaled, unscaled_param
        else:
            # Return scaled embeddings and unscaled parameters when not in training mode
            embeddings = (unscaled_param.cuda() - torch.tensor(self.scaler.mean_).cuda()) / \
                         torch.tensor(self.scaler.var_ ** 0.5).cuda()
            return out_scaled, embeddings, unscaled_param
