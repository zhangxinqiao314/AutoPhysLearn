import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from m3util.ml.inference import computeTime
from m3util.util.IO import make_folder
from m3util.ml.rand import set_seeds
from m3util.ml.logging import write_csv, save_list_to_txt
from m3util.ml.optimizers.AdaHessian import AdaHessian
from m3util.ml.optimizers.TrustRegion import TRCG
from datafed_torchflow.pytorch import TorchLogger

import numpy as np


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

    def __init__(
        self,
        function,
        x_data,
        input_channels,
        num_params,
        scaler=None,
        post_processing=None,
        device="cuda",
        loops_scaler=None,
        **kwargs,
    ):
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
            nn.AdaptiveAvgPool1d(
                64
            ),  # Adaptive average pooling to reduce output dimensionality
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
                embedding * torch.tensor(self.scaler.var_**0.5).cuda()
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
            out_scaled = (
                out - torch.tensor(self.loops_scaler.mean).cuda()
            ) / torch.tensor(self.loops_scaler.std).cuda()
        else:
            out_scaled = out

        if self.training:
            return out_scaled, unscaled_param
        else:
            # Return scaled embeddings and unscaled parameters when not in training mode
            embeddings = (
                unscaled_param.cuda() - torch.tensor(self.scaler.mean_).cuda()
            ) / torch.tensor(self.scaler.var_**0.5).cuda()
            return out_scaled, embeddings, unscaled_param


class Model(nn.Module):
    """
    A wrapper model class for training, evaluating, and predicting with a neural network model.
    Provides functionality for early stopping, saving models, and computing inference time.

    Attributes:
        model (nn.Module): The neural network model to be trained and evaluated.
        dataset (Dataset): The dataset used by the model.
        model_basename (str): Base name for saving the model checkpoints.
        training (bool): Flag indicating whether the model is in training mode.
        path (str): Path where trained models and logs are saved.
        device (str): Device to run the computations on ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model,
        dataset,
        model_basename="",
        training=True,
        path="Trained Models/SHO Fitter/",
        device=None,
        datafed_path=None,
        script_path=None,
        **kwargs,
    ):
        """
        Initializes the Model class.

        Args:
            model (nn.Module): The neural network model to be wrapped.
            dataset (Dataset): The dataset associated with the model.
            model_basename (str, optional): Base name for saving model checkpoints. Defaults to ''.
            training (bool, optional): Flag indicating if the model is in training mode. Defaults to True.
            path (str, optional): Path to save trained models. Defaults to 'Trained Models/SHO Fitter/'.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to None.
            datafed_path (str, optional): Path to save models in DataFed. If the path is None, it will not save to DataFed. Defaults to None.
            script_path (str, optional): Path to the script that is being run. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        # Set the device to CUDA if available, otherwise use CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"Using GPU {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("Using CPU")
        else:
            self.device = device

        self.model = model
        self.model.dataset = dataset
        self.model.training = True
        self.model_name = model_basename
        self.path = make_folder(path)
        self.datafed_path = datafed_path
        self.script_path = script_path
        
        # Checks if the user wants to save the data to DataFed.
        if self.datafed_path is not None:
            self.datafed = True
        else: 
            self.datafed = False

    def select_optimizer(self, optimizer, **kwargs):
        # Select the optimizer based on the provided input
        if optimizer == "Adam":
            optimizer_ = torch.optim.Adam(self.model.parameters())
        elif optimizer == "AdaHessian":
            optimizer_ = AdaHessian(self.model.parameters(), lr=0.5)
        elif isinstance(optimizer, dict) and optimizer["name"] == "TRCG":
            optimizer_ = optimizer["optimizer"](
                self.model, optimizer["radius"], optimizer["device"]
            )
        else:
            try:
                optimizer_ = optimizer(self.model.parameters())
            except:
                raise ValueError("Optimizer not recognized")
        return optimizer_

    def extract_kwargs(
        self,
        i,
        model,
        optimizer_name,
        epoch,
        total_time,
        train_loss,
        total_num,
        batch_size,
        loss_func,
        seed,
        stopping_early,
        model_updates,
        file_name,
    ):
        """
        Extracts the provided values and returns them as keyword arguments.

        Args:
            i (int or None): Training index, exclude if None.
            model (object): The model object containing the dataset and noise level.
            optimizer_name (str): The name of the optimizer used.
            epoch (int): Current epoch number.
            total_time (float): Total training time.
            train_loss (float): Total training loss.
            total_num (int): The total number of training examples.
            batch_size (int): Size of each training batch.
            loss_func (str): Loss function used.
            seed (int): Seed used for training.
            model_updates (int): Number of mini-batches completed.
            file_name (str): Name of the file to save.

        Returns:
            dict: A dictionary containing the extracted keyword arguments.
        """

        kwargs = {
            "noise_level": self.model.dataset.noise,
            "optimizer_name": optimizer_name,
            "epoch": epoch,
            "total_time": total_time,
            "train_loss": train_loss,
            "batch_size": batch_size,
            "loss_func": loss_func,
            "seed": seed,
            "early_stopping": stopping_early,
            "model_updates": model_updates,
            "file_name": file_name,
        }

        # Only include 'training_index' if i is not None
        if i is not None:
            kwargs["training_index"] = i

        return kwargs

    def fit(
        self,
        data_train,
        batch_size=200,
        epochs=5,
        loss_func=torch.nn.MSELoss(),
        optimizer="Adam",
        seed=42,
        datatype=torch.float32,
        save_all=False,
        write_CSV=None,
        closure=None,
        basepath=None,
        early_stopping_loss=None,
        early_stopping_count=None,
        early_stopping_time=None,
        save_training_loss=True,
        i=None,
        **kwargs,
    ):
        """
        Trains the model on the provided training data.

        Args:
            data_train (Dataset): Training dataset.
            batch_size (int, optional): Batch size for training. Defaults to 200.
            epochs (int, optional): Number of epochs to train. Defaults to 5.
            loss_func (callable, optional): Loss function to use. Defaults to torch.nn.MSELoss().
            optimizer (str or dict, optional): Optimizer to use. Can be 'Adam', 'AdaHessian', or a custom optimizer. Defaults to 'Adam'.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            datatype (torch.dtype, optional): Data type for training. Defaults to torch.float32.
            save_all (bool, optional): Whether to save the model at every epoch. Defaults to False.
            write_CSV (str, optional): Path to write CSV logs. Defaults to None.
            closure (callable, optional): Custom closure function for optimizers like TRCG. Defaults to None.
            basepath (str, optional): Base path for saving models. Defaults to None.
            early_stopping_loss (float, optional): Early stopping threshold for loss. Defaults to None.
            early_stopping_count (int, optional): Early stopping patience count. Defaults to None.
            early_stopping_time (float, optional): Early stopping time threshold. Defaults to None.
            save_training_loss (bool, optional): Whether to save the training loss. Defaults to True.
            i (int, optional): Index for logging. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        loss_ = []

        # Set the path for saving models
        if basepath is not None:
            path = f"{self.path}/{basepath}/"
            make_folder(path)
            print(f"Saving to {path}")
        else:
            path = self.path

        # Set the model to the specified datatype and device
        self.to(datatype).to(self.device)

        # Set the random seed for reproducibility
        set_seeds(seed=seed)

        # Clear the GPU cache
        torch.cuda.empty_cache()

        # Select the optimizer
        optimizer_ = self.select_optimizer(optimizer, **kwargs)

        # Instantiate the dataloader
        train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

        # If using Trust Region CG, store the TR optimizer and instantiate the Adam optimizer
        if isinstance(optimizer_, TRCG):
            TRCG_OP = optimizer_
            optimizer_ = torch.optim.Adam(self.model.parameters(), **kwargs)

        # Initialize variables for early stopping
        total_time = 0
        low_loss_count = 0
        already_stopped = False  # Flag for early stopping
        model_updates = 0

        if self.datafed_path is None:
            self.datafed = False

        torchlogger = TorchLogger(
            self.model,
            self.datafed_path,
            script_path=self.script_path,
            local_path=self.path,
        )
        
        # saves the notebook record id to the torchlogger object
        if torchlogger.notebook_record_id is not None:
            self.notebook_record_id = torchlogger.notebook_record_id

        # Training loop over epochs
        for epoch in range(epochs):
            train_loss = 0.0
            total_num = 0
            epoch_time = 0

            # Set the model to training mode
            self.model.train()

            # Iterate over batches in the dataloader
            for train_batch in train_dataloader:
                model_updates += 1
                start_time = time.time()

                # Move the batch to the correct datatype and device
                train_batch = train_batch.to(datatype).to(self.device)

                if "TRCG_OP" in locals() and epoch > optimizer.get("ADAM_epochs", -1):

                    def closure(part, total, device):
                        pred, embedding = self.model(train_batch)
                        pred = pred.to(torch.float32)
                        pred = torch.atleast_3d(pred)
                        embedding = embedding.to(torch.float32)
                        loss = loss_func(train_batch, pred)
                        return loss

                    # Perform a Trust Region CG step
                    loss, radius, cnt_compute, cg_iter = TRCG_OP.step(closure)
                    train_loss += loss * train_batch.shape[0]
                    total_num += train_batch.shape[0]
                    optimizer_name = "Trust Region CG"
                else:
                    # Standard optimizer step (e.g., Adam)
                    pred, embedding = self.model(train_batch)
                    pred = pred.to(torch.float32)
                    pred = torch.atleast_3d(pred)
                    embedding = embedding.to(torch.float32)
                    optimizer_.zero_grad()
                    loss = loss_func(train_batch, pred)
                    loss.backward(create_graph=True)
                    train_loss += loss.item() * pred.shape[0]
                    total_num += pred.shape[0]
                    optimizer_.step()
                    for param in self.model.parameters():
                        param.grad = None
                    optimizer_name = type(optimizer_).__name__

                epoch_time += time.time() - start_time
                total_time += time.time() - start_time

                # Store the loss for logging
                try:
                    loss_.append(loss.item())
                except:
                    loss_.append(loss)

                # sets the optimizer in the torchlogger object
                torchlogger.optimizer = optimizer_

                # Early stopping based on loss
                if early_stopping_loss is not None and not already_stopped:
                    if loss < early_stopping_loss:
                        low_loss_count += train_batch.shape[0]
                        if low_loss_count >= early_stopping_count:
                            filename = f"Early_Stoppage_at_{total_time}_{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss/total_num}.pth"

                            datafed_kwargs = self.extract_kwargs(
                                i,
                                self.model_name,
                                optimizer_name,
                                epoch,
                                total_time,
                                train_loss,
                                total_num,
                                batch_size,
                                loss_func,
                                seed,
                                True,
                                model_updates,
                                file_name=filename,
                            )

                            torchlogger.save(
                                filename, datafed=self.datafed, **datafed_kwargs
                            )

                            write_csv(
                                write_CSV,
                                path,
                                self.model_name,
                                i,  # training index exclude if None
                                self.model.dataset.noise,  # noise level
                                optimizer_name,  # optimizer name
                                epoch,  # epoch
                                total_time,  # total training time
                                train_loss / total_num,  # train loss
                                batch_size,  # batch size
                                loss_func,  # loss function
                                seed,  # Training Seed
                                True,  # early stopping
                                model_updates,  # number of mini-batches completed
                            )

                            already_stopped = True
                    else:
                        low_loss_count -= train_batch.shape[0] * 5

            # Verbose logging
            if kwargs.get("verbose", False):
                print(f"Loss = {loss.item()}")

            train_loss /= total_num

            print(f"{optimizer_name}")
            print(f"epoch : {epoch+1}/{epochs}, recon loss = {train_loss:.8f}")
            print(f"--- {epoch_time} seconds ---")

            # Print the current learning rate (optional)
            current_lr = optimizer_.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}, Learning Rate: {current_lr}")

            # Save the model at each epoch if save_all is True
            if save_all:
                filename = f"{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss}.pth"

                datafed_kwargs = self.extract_kwargs(
                    i,
                    self.model_name,
                    optimizer_name,
                    epoch,
                    total_time,
                    train_loss,
                    total_num,
                    batch_size,
                    loss_func,
                    seed,
                    False,
                    model_updates,
                    file_name=filename,
                )

                torchlogger.save(filename, datafed=self.datafed, **datafed_kwargs)

            # Early stopping based on time
            if early_stopping_time is not None:
                if total_time > early_stopping_time:
                    filename = f"Early_Stoppage_at_{total_time}_{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss}.pth"

                    datafed_kwargs = self.extract_kwargs(
                        i,
                        self.model_name,
                        optimizer_name,
                        epoch,
                        total_time,
                        train_loss,
                        total_num,
                        batch_size,
                        loss_func,
                        seed,
                        True,
                        model_updates,
                        file_name=filename,
                    )

                    torchlogger.save(filename, datafed=self.datafed, **datafed_kwargs)

                    write_csv(
                        write_CSV,
                        path,
                        self.model_name,
                        i,
                        self.model.dataset.noise,
                        optimizer_name,
                        epoch,
                        total_time,
                        train_loss,  # already divided by total_num
                        batch_size,
                        loss_func,
                        seed,
                        True,
                        model_updates,
                    )
                    break

        # Save the final model
        filename = f"{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss}.pth"

        datafed_kwargs = self.extract_kwargs(
            i,
            self.model_name,
            optimizer_name,
            epoch,
            total_time,
            train_loss,
            total_num,
            batch_size,
            loss_func,
            seed,
            False,
            model_updates,
            file_name=filename,
        )

        torchlogger.save(filename, datafed=self.datafed, **datafed_kwargs)

        write_csv(
            write_CSV,
            path,
            self.model_name,
            i,
            self.model.dataset.noise,
            optimizer_name,
            epoch,
            total_time,
            train_loss,  # already divided by total_num
            batch_size,
            loss_func,
            seed,
            False,
            model_updates,
        )

        # Save training loss if required
        if save_training_loss:
            save_list_to_txt(
                loss_,
                f"{path}/Training_loss_{self.model_name}_model_optimizer_{optimizer_name}_epoch_{epoch}_train_loss_{train_loss}.txt",
            )

        # Set model to evaluation mode after training
        self.model.eval()

    def load(self, model_path):
        """
        Loads a saved model state from the specified path.

        Args:
            model_path (str): Path to the saved model state file.
        """
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def inference_timer(self, data, batch_size=0.5e4):
        """
        Measures the inference time for the model on the given data.

        Args:
            data (Dataset): Dataset for inference.
            batch_size (float, optional): Batch size for inference. Defaults to .5e4.
        """
        torch.cuda.empty_cache()

        batch_size = int(batch_size)

        dataloader = DataLoader(data, batch_size)

        # Computes and prints the inference time
        computeTime(self.model, dataloader, batch_size, device=self.device)

    def predict(
        self, data, batch_size=10000, single=False, translate_params=True, is_SHO=True
    ):
        """
        Generates predictions for the given data using the trained model.

        Args:
            data (Dataset): Dataset for predictions.
            batch_size (int, optional): Batch size for predictions. Defaults to 10000.
            single (bool, optional): Flag for single sample prediction. Defaults to False.
            translate_params (bool, optional): Whether to translate parameters (e.g., phase correction). Defaults to True.
            is_SHO (bool, optional): Flag indicating if the data corresponds to SHO (Simple Harmonic Oscillator) fits. Defaults to True.

        Returns:
            tuple: Predictions, scaled parameters, and unscaled parameters.
        """
        self.model.eval()

        dataloader = DataLoader(data, batch_size=batch_size)

        # Preallocate tensors for predictions and parameters
        num_elements = len(dataloader.dataset)
        num_batches = len(dataloader)
        data = data.clone().detach().requires_grad_(True)
        predictions = torch.zeros_like(data.clone().detach())
        params_scaled = torch.zeros((data.shape[0], self.model.num_params))
        params = torch.zeros((data.shape[0], self.model.num_params))

        # Compute predictions for each batch
        for i, train_batch in enumerate(dataloader):
            start = i * batch_size
            end = start + batch_size

            if i == num_batches - 1:
                end = num_elements

            pred_batch, params_scaled_, params_ = self.model(
                train_batch.to(self.device)
            )

            if is_SHO:
                predictions[start:end] = pred_batch.cpu().detach()
            else:
                predictions[start:end] = torch.unsqueeze(pred_batch.cpu().detach(), 2)
            params_scaled[start:end] = params_scaled_.cpu().detach()
            params[start:end] = params_.cpu().detach()

            torch.cuda.empty_cache()

        # Translate parameters if required (e.g., phase correction)
        if translate_params:
            params[params[:, 0] < 0, 3] = params[params[:, 0] < 0, 3] - np.pi
            params[params[:, 0] < 0, 0] = np.abs(params[params[:, 0] < 0, 0])

        # Apply phase shift correction if needed
        if self.model.dataset.NN_phase_shift is not None:
            params_scaled[:, 3] = torch.Tensor(
                self.model.dataset.shift_phase(
                    params_scaled[:, 3].detach().numpy(),
                    self.model.dataset.NN_phase_shift,
                )
            )
            params[:, 3] = torch.Tensor(
                self.model.dataset.shift_phase(
                    params[:, 3].detach().numpy(), self.model.dataset.NN_phase_shift
                )
            )

        return predictions, params_scaled, params
