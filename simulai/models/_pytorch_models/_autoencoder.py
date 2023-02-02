# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import torch
import numpy as np
from typing import Union, Tuple, Optional

from simulai.templates import NetworkTemplate, as_tensor
from simulai.regression import DenseNetwork
from simulai.regression import ConvolutionalNetwork, Linear

########################################
### Some usual AutoEncoder architectures
########################################

class AutoencoderMLP(NetworkTemplate):
    """
     This is an implementation of a Fully-connected AutoEncoder as
           Reduced Order Model;

               A MLP autoencoder architecture consists of two stages:
               --> Fully-connected encoder
               --> Fully connected decoder

           SCHEME:
                   |         |
                   |  |   |  |
           Z ->    |  | | |  |  -> Z_til
                   |  |   |  |
                   |         |

              ENCODER       DECODER
    """
    def __init__(self,
                 encoder: DenseNetwork = None,
                 decoder: DenseNetwork = None,
                 input_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 latent_dim: Optional[int] = None,
                 activation: Optional[Union[list, str]] = None,
                 devices: Union[str, list] = 'cpu') -> None:

        super(AutoencoderMLP, self).__init__()

        self.weights = list()

        # This option is used when no network is provided
        # and it uses default choices for the architectures
        if encoder == None and decoder == None:

            from simulai.templates import mlp_autoencoder_auto

            encoder, decoder = mlp_autoencoder_auto(input_dim=input_dim,
                                                    latent_dim=latent_dim,
                                                    output_dim=output_dim,
                                                    activation=activation)

        # Determining the kind of device to be used for allocating the
        # subnetworks used in the DeepONet model
        self.device = self._set_device(devices=devices)

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.add_module('encoder', self.encoder)
        self.add_module('decoder', self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.decoder.weights

        self.last_encoder_channels = None


    def summary(self) -> None:
        """

        It prints the summary of the network architecture

        """

        self.encoder.summary()
        self.decoder.summary()

    def projection(
            self,
            input_data: Union[np.ndarray,
                              torch.Tensor] = None) -> torch.Tensor:
        """

        Projecting the input dataset into the latent space

        :param input_data: the dataset to be projected
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset projected over the latent space
        :rtype: torch.Tensor

        """

        latent = self.encoder.forward(input_data=input_data)

        return latent

    def reconstruction(
            self,
            input_data: Union[torch.Tensor,
                              np.ndarray] = None) -> torch.Tensor:
        """

        Reconstructing the latent dataset to the original one

        :param input_data: the dataset to be reconstructed
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset reconstructed
        :rtype: torch.Tensor

        """

        reconstructed = self.decoder.forward(input_data=input_data)

        return reconstructed

    def forward(
            self,
            input_data: Union[np.ndarray,
                              torch.Tensor] = None) -> torch.Tensor:
        """

        Executing the complete projection/reconstruction pipeline

        :param input_data: the input dataset
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset reconstructed
        :rtype: torch.Tensor

        """

        latent = self.projection(input_data=input_data)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed

    def eval_projection(
            self,
            input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        """

        Projecting the input dataset into the latent space

        :param input_data: the dataset to be projected
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset projected over the latent space
        :rtype: np.ndarray

        """

        return self.projection(input_data=input_data).detach().numpy()

# Convolutional AutoEncoder
class AutoencoderCNN(NetworkTemplate):
    r"""
    This is an implementation of a convolutional autoencoder as Reduced Order Model.
    An autoencoder architecture consists of three stages:
    
    * The convolutional encoder
    
    The bottleneck stage, subdivided in:
        * Fully-connected encoder
        * Fully connected decoder
        * The convolutional decoder

    SCHEME:
    
    Z -> [Conv] -> [Conv] -> ... [Conv] -> |  | | |  | -> [Conv.T] -> [Conv.T] -> ... [Conv.T] -> Z_til


                    ENCODER               DENSE BOTTLENECK           DECODER
    """
    def __init__(self,
                 encoder: ConvolutionalNetwork = None,
                 bottleneck_encoder: Linear = None,
                 bottleneck_decoder: Linear = None,
                 decoder: ConvolutionalNetwork = None,
                 encoder_activation: str = 'relu',
                 input_dim: Optional[Tuple[int, ...]] = None,
                 output_dim: Optional[Tuple[int, ...]] = None,
                 latent_dim: Optional[int] = None,
                 activation: Optional[Union[list, str]] = None,
                 channels: Optional[int] = None,
                 case: Optional[str] = None,
                 devices: Union[str, list] = 'cpu') -> None:

        super(AutoencoderCNN, self).__init__()

        self.weights = list()

        # Determining the kind of device to be used for allocating the
        # subnetworks
        self.device = self._set_device(devices=devices)

        # If not network is provided, the automatic generation
        # pipeline is activated.
        if all([isn == None for isn in [encoder, decoder,
                                        bottleneck_encoder, bottleneck_decoder]]):

            from simulai.templates import cnn_autoencoder_auto

            encoder, decoder,\
            bottleneck_encoder,\
            bottleneck_decoder = cnn_autoencoder_auto(input_dim=input_dim,
                                                      latent_dim=latent_dim,
                                                      output_dim=output_dim,
                                                      activation=activation,
                                                      channels=channels,
                                                      case=case)

        self.encoder = encoder.to(self.device)
        self.bottleneck_encoder = bottleneck_encoder.to(self.device)
        self.bottleneck_decoder = bottleneck_decoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.add_module('encoder', self.encoder)
        self.add_module('bottleneck_encoder', self.bottleneck_encoder)
        self.add_module('bottleneck_decoder', self.bottleneck_decoder)
        self.add_module('decoder', self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.bottleneck_encoder.weights
        self.weights += self.bottleneck_decoder.weights
        self.weights += self.decoder.weights

        self.last_encoder_channels = None
        self.before_flatten_dimension = None

        self.encoder_activation = self._get_operation(
            operation=encoder_activation)

    def summary(self,
                input_data: Union[np.ndarray, torch.Tensor] = None,
                input_shape: list = None) -> torch.Tensor:
        """

        It prints the summary of the network architecture

        :param input_data: the input dataset
        :type input_data: Union[np.ndarray, torch.Tensor]
        :param input_shape: the shape of the input data
        :type input_shape: list
        :returns: the dataset projected over the latent space
        :rtype: torch.Tensor

        """

        self.encoder.summary(input_data=input_data,
                             input_shape=input_shape,
                             device=self.device)

        if isinstance(input_data, np.ndarray):
            btnk_input = self.encoder.forward(input_data=input_data)
        else:
            assert input_shape, "It is necessary to have input_shape when input_data is None."
            input_shape = self.encoder.input_size
            input_shape[0] = 1

            input_data = torch.ones(input_shape).to(self.device)

            btnk_input = self.encoder.forward(input_data=input_data)

        before_flatten_dimension = tuple(btnk_input.shape[1:])
        btnk_input = btnk_input.reshape((-1, np.prod(btnk_input.shape[1:])))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)

        self.bottleneck_encoder.summary()
        self.bottleneck_decoder.summary()

        bottleneck_output = self.encoder_activation(
            self.bottleneck_decoder.forward(input_data=latent))

        bottleneck_output = bottleneck_output.reshape(
            (-1, *before_flatten_dimension))

        self.decoder.summary(input_data=bottleneck_output, device=self.device)

    @as_tensor
    def projection(
            self, input_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """

        Projecting the input dataset into the latent space

        :param input_data: the dataset to be projected
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset projected over the latent space
        :rtype: torch.Tensor

        """

        btnk_input = self.encoder.forward(input_data=input_data)

        self.before_flatten_dimension = tuple(btnk_input.shape[1:])

        btnk_input = btnk_input.reshape(
            (-1, np.prod(self.before_flatten_dimension)))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)

        return latent

    @as_tensor
    def reconstruction(
            self, input_data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """

        Reconstructing the latent dataset to the original one

        :param input_data: the dataset to be reconstructed
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset reconstructed
        :rtype: torch.Tensor

        """

        bottleneck_output = self.encoder_activation(
            self.bottleneck_decoder.forward(input_data=input_data))

        bottleneck_output = bottleneck_output.reshape(
            (-1, ) + self.before_flatten_dimension)

        reconstructed = self.decoder.forward(input_data=bottleneck_output)

        return reconstructed

    def forward(self, input_data: Union[np.ndarray,
                                        torch.Tensor]) -> torch.Tensor:
        """

        Executing the complete projection/reconstruction pipeline
        :param input_data: the input dataset
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset reconstructed
        :rtype: torch.Tensor

        """

        latent = self.projection(input_data=input_data)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed

    def eval(self,
             input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        """

        Projecting the input dataset into the latent space

        :param input_data: the dataset to be projected
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset projected over the latent space
        :rtype: np.ndarray

        """

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype('float32'))

        input_data = input_data.to(self.device)

        return super().eval(input_data=input_data)

    def project(
            self,
            input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        """

        Projecting the input dataset into the latent space
        :param input_data: the dataset to be projected
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset projected over the latent space
        :rtype: np.ndarray

        """

        projected_data = self.projection(input_data=input_data)

        return projected_data.cpu().detach().numpy()

    def reconstruct(
            self,
            input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        """

        Reconstructing the latent dataset to the original one

        :param input_data: the dataset to be reconstructed
        :type input_data: Union[np.ndarray, torch.Tensor]
        :returns: the dataset reconstructed
        :rtype: np.ndarray

        """

        reconstructed_data = self.reconstruction(input_data=input_data)

        return reconstructed_data.cpu().detach().numpy()


class AutoencoderKoopman(NetworkTemplate):
    """

    This is an implementation of a Koopman autoencoder as

    Reduced Order Model:
    --------------------
    A Koopman autoencoder architecture consists of five stages:
    --> The convolutional encoder [Optional]
    --> Fully-connected encoder
    --> Koopman operator
    --> Fully connected decoder
    --> The convolutional decoder [Optional]

    SCHEME:
                                         (Koopman OPERATOR)
                                                  ^
                                           |      |      |
                                           |  |   |   |  |
    Z -> [Conv] -> [Conv] -> ... [Conv] -> |  | | - | |  | -> [Conv.T] -> [Conv.T] -> ... [Conv.T] -> Z_til
                                           |  |       |  |
                                           |             |

                    ENCODER               DENSE BOTTLENECK           DECODER

    """
    def __init__(self,
                 encoder: Union[ConvolutionalNetwork, DenseNetwork] = None,
                 bottleneck_encoder: Optional[Union[Linear,
                                                    DenseNetwork]] = None,
                 bottleneck_decoder: Optional[Union[Linear,
                                                    DenseNetwork]] = None,
                 decoder: Union[ConvolutionalNetwork, DenseNetwork] = None,
                 input_dim: Optional[Tuple[int, ...]] = None,
                 output_dim: Optional[Tuple[int, ...]] = None,
                 latent_dim: Optional[int] = None,
                 activation: Optional[Union[list, str]] = None,
                 channels: Optional[int] = None,
                 case: Optional[str] = None,
                 architecture: Optional[str] = None,
                 encoder_activation: str = 'relu',
                 devices: Union[str, list] = 'cpu') -> None:

        super(AutoencoderKoopman, self).__init__()

        self.weights = list()

        # Determining the kind of device to be used for allocating the
        # subnetworks
        self.device = self._set_device(devices=devices)

        # If not network is provided, the automatic generation
        # pipeline is activated.
        if all([isn == None for isn in [encoder, decoder,
                                        bottleneck_encoder, bottleneck_decoder]]):

            from simulai.templates import autoencoder_auto

            encoder, decoder, \
            bottleneck_encoder, \
            bottleneck_decoder = autoencoder_auto(input_dim=input_dim,
                                                  latent_dim=latent_dim,
                                                  output_dim=output_dim,
                                                  activation=activation,
                                                  channels=channels,
                                                  architecture=architecture,
                                                  case=case)

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.add_module('encoder', self.encoder)
        self.add_module('decoder', self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.decoder.weights

        # These subnetworks are optional
        if bottleneck_encoder is not None and bottleneck_decoder is not None:
            self.bottleneck_encoder = bottleneck_encoder.to(self.device)
            self.bottleneck_decoder = bottleneck_decoder.to(self.device)

            self.add_module('bottleneck_encoder', self.bottleneck_encoder)
            self.add_module('bottleneck_decoder', self.bottleneck_decoder)

            self.weights += self.bottleneck_encoder.weights
            self.weights += self.bottleneck_decoder.weights

        # These subnetworks are optional
        if bottleneck_encoder is not None and bottleneck_decoder is not None:
            self.bottleneck_encoder = bottleneck_encoder.to(self.device)
            self.bottleneck_decoder = bottleneck_decoder.to(self.device)

            self.add_module('bottleneck_encoder', self.bottleneck_encoder)
            self.add_module('bottleneck_decoder', self.bottleneck_decoder)

            self.weights += self.bottleneck_encoder.weights
            self.weights += self.bottleneck_decoder.weights

        if bottleneck_encoder is not None and bottleneck_decoder is not None:

            self.projection = self._projection_with_bottleneck
            self.reconstruction = self._reconstruction_with_bottleneck
        else:
            self.projection = self._projection
            self.reconstruction = self._reconstruction

        self.last_encoder_channels = None
        self.before_flatten_dimension = None

        self.latent_dimension = None

        if bottleneck_encoder is not None:
            self.latent_dimension = bottleneck_encoder.output_size
        else:
            self.latent_dimension = self.encoder.output_size

        self.K_op = torch.nn.Linear(self.latent_dimension,
                                    self.latent_dimension,
                                    bias=False).weight.to(self.device)

        self.encoder_activation = self._get_operation(
            operation=encoder_activation)

    def summary(self,
                input_data: Union[np.ndarray, torch.Tensor] = None,
                input_shape: list = None) -> torch.Tensor:

        self.encoder.summary(input_data=input_data,
                             input_shape=input_shape,
                             device=self.device)

        self.bottleneck_encoder.summary()
        self.bottleneck_decoder.summary()

    @as_tensor
    def _projection_with_bottleneck(
            self,
            input_data: Union[np.ndarray,
                              torch.Tensor] = None) -> torch.Tensor:

        btnk_input = self.encoder.forward(input_data=input_data)

        self.before_flatten_dimension = tuple(btnk_input.shape[1:])

        btnk_input = btnk_input.reshape(
            (-1, np.prod(self.before_flatten_dimension)))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)

        return latent

    @as_tensor
    def _projection(
            self,
            input_data: Union[np.ndarray,
                              torch.Tensor] = None) -> torch.Tensor:

        latent = self.encoder.forward(input_data=input_data)

        return latent

    @as_tensor
    def _reconstruction_with_bottleneck(
            self,
            input_data: Union[torch.Tensor,
                              np.ndarray] = None) -> torch.Tensor:

        bottleneck_output = self.encoder_activation(
            self.bottleneck_decoder.forward(input_data=input_data))

        bottleneck_output = bottleneck_output.reshape(
            (-1, ) + self.before_flatten_dimension)

        reconstructed = self.decoder.forward(input_data=bottleneck_output)

        return reconstructed

    @as_tensor
    def _reconstruction(
            self,
            input_data: Union[torch.Tensor,
                              np.ndarray] = None) -> torch.Tensor:

        reconstructed = self.decoder.forward(input_data=input_data)

        return reconstructed

    # Evaluating the operation u^{u+m} = K^m u^{i}
    def latent_forward_m(self,
                         input_data: Union[np.ndarray, torch.Tensor] = None,
                         m: int = 1) -> torch.Tensor:

        return torch.matmul(input_data, torch.pow(self.K_op.T, m))

    # Evaluating the operation u^{u+1} = K u^{i}
    def latent_forward(
            self,
            input_data: Union[np.ndarray,
                              torch.Tensor] = None) -> torch.Tensor:

        return torch.matmul(input_data, self.K_op.T)

    # Evaluating the operation Ũ = D(E(U))
    def reconstruction_forward(
            self,
            input_data: Union[np.ndarray,
                              torch.Tensor] = None) -> torch.Tensor:

        latent = self.projection(input_data=input_data)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed

    # Evaluating the operation Ũ_m = D(K^m E(U))
    def reconstruction_forward_m(self,
                                 input_data: Union[np.ndarray,
                                                   torch.Tensor] = None,
                                 m: int = 1) -> torch.Tensor:

        latent = self.projection(input_data=input_data)
        latent_m = self.latent_forward_m(input_data=latent, m=m)
        reconstructed_m = self.reconstruction(input_data=latent_m)

        return reconstructed_m

    def predict(self,
                input_data: Union[np.ndarray, torch.Tensor] = None,
                n_steps: int = 1) -> np.ndarray:

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype('float32'))

        predictions = list()
        latent = self.projection(input_data=input_data)
        init_latent = latent

        # Extrapolating in the latent space over n_steps steps
        for s in range(n_steps):
            latent_s = self.latent_forward(input_data=init_latent)
            init_latent = latent_s
            predictions.append(latent_s)

        predictions = torch.vstack(predictions)

        reconstructed_predictions = self.reconstruction(input_data=predictions)

        return reconstructed_predictions.detach().numpy()

    def project(
            self,
            input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:

        projected_data = self.projection(input_data=input_data)

        return projected_data.cpu().detach().numpy()

    def reconstruct(
            self,
            input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:

        reconstructed_data = self.reconstruction(input_data=input_data)

        return reconstructed_data.cpu().detach().numpy()

class AutoencoderVariational(NetworkTemplate):
    """

           This is an implementation of a Koopman autoencoder as
           Reduced Order Model;

               A Koopman autoencoder architecture consists of five stages:
               --> The convolutional encoder [Optional]
               --> Fully-connected encoder
               --> Gaussian noise
               --> Fully connected decoder
               --> The convolutional decoder [Optional]

           SCHEME:
                                                         Gaussian noise
                                                         ^
                                                  |      |      |
                                                  |  |   |   |  |
           Z -> [Conv] -> [Conv] -> ... [Conv] -> |  | | - | |  | -> [Conv.T] -> [Conv.T] -> ... [Conv.T] -> Z_til
                                                  |  |       |  |
                                                  |             |

                          ENCODER               DENSE BOTTLENECK           DECODER

    """
    def __init__(self,
                 encoder: Union[ConvolutionalNetwork, DenseNetwork] = None,
                 bottleneck_encoder: Optional[Union[Linear,
                                                    DenseNetwork]] = None,
                 bottleneck_decoder: Optional[Union[Linear,
                                                    DenseNetwork]] = None,
                 decoder: Union[ConvolutionalNetwork, DenseNetwork] = None,
                 encoder_activation: str = 'relu',
                 input_dim: Optional[Tuple[int, ...]] = None,
                 output_dim: Optional[Tuple[int, ...]] = None,
                 latent_dim: Optional[int] = None,
                 activation: Optional[Union[list, str]] = None,
                 channels: Optional[int] = None,
                 case: Optional[str] = None,
                 architecture: Optional[str] = None,
                 scale: float = 1e-3,
                 devices: Union[str, list] = 'cpu') -> None:

        super(AutoencoderVariational, self).__init__()

        self.weights = list()

        # Determining the kind of device to be used for allocating the
        # subnetworks
        self.device = self._set_device(devices=devices)

        # If not network is provided, the automatic generation
        # pipeline is activated.
        if all([isn == None for isn in [encoder, decoder,
                                        bottleneck_encoder, bottleneck_decoder]]):

            from simulai.templates import autoencoder_auto

            encoder, decoder, \
            bottleneck_encoder, \
            bottleneck_decoder = autoencoder_auto(input_dim=input_dim,
                                                  latent_dim=latent_dim,
                                                  output_dim=output_dim,
                                                  activation=activation,
                                                  channels=channels,
                                                  architecture=architecture,
                                                  case=case)

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.add_module('encoder', self.encoder)
        self.add_module('decoder', self.decoder)

        self.weights += self.encoder.weights
        self.weights += self.decoder.weights

        # These subnetworks are optional
        if bottleneck_encoder is not None and bottleneck_decoder is not None:
            self.bottleneck_encoder = bottleneck_encoder.to(self.device)
            self.bottleneck_decoder = bottleneck_decoder.to(self.device)

            self.add_module('bottleneck_encoder', self.bottleneck_encoder)
            self.add_module('bottleneck_decoder', self.bottleneck_decoder)

            self.weights += self.bottleneck_encoder.weights
            self.weights += self.bottleneck_decoder.weights

        if bottleneck_encoder is not None and bottleneck_decoder is not None:

            self.projection = self._projection_with_bottleneck
            self.reconstruction = self._reconstruction_with_bottleneck
        else:
            self.projection = self._projection
            self.reconstruction = self._reconstruction

        self.last_encoder_channels = None
        self.before_flatten_dimension = None

        self.latent_dimension = None

        if bottleneck_encoder is not None:
            self.latent_dimension = bottleneck_encoder.output_size
        else:
            self.latent_dimension = self.encoder.output_size

        self.z_mean = torch.nn.Linear(self.latent_dimension,
                                      self.latent_dimension).to(self.device)
        self.z_log_var = torch.nn.Linear(self.latent_dimension,
                                         self.latent_dimension).to(self.device)

        self.add_module('z_mean', self.z_mean)
        self.add_module('z_log_var', self.z_log_var)

        self.weights += [self.z_mean.weight]
        self.weights += [self.z_log_var.weight]

        self.mu = None
        self.log_v = None
        self.scale = scale

        self.encoder_activation = self._get_operation(
            operation=encoder_activation)

    def summary(self,
                input_data: Union[np.ndarray, torch.Tensor] = None,
                input_shape: list = None) -> torch.Tensor:

        self.encoder.summary(input_data=input_data,
                             input_shape=input_shape,
                             device=self.device)

        self.before_flatten_dimension = tuple(self.encoder.output_size[1:])

        self.bottleneck_encoder.summary()
        self.bottleneck_decoder.summary()

    @as_tensor
    def _projection_with_bottleneck(
            self,
            input_data: Union[np.ndarray,
                              torch.Tensor] = None) -> torch.Tensor:

        btnk_input = self.encoder.forward(input_data=input_data)

        self.before_flatten_dimension = tuple(self.encoder.output_size[1:])

        btnk_input = btnk_input.reshape(
            (-1, np.prod(self.before_flatten_dimension)))

        latent = self.bottleneck_encoder.forward(input_data=btnk_input)

        return latent

    @as_tensor
    def _projection(
            self,
            input_data: Union[np.ndarray,
                              torch.Tensor] = None) -> torch.Tensor:

        latent = self.encoder.forward(input_data=input_data)

        return latent

    @as_tensor
    def _reconstruction_with_bottleneck(
            self,
            input_data: Union[torch.Tensor,
                              np.ndarray] = None) -> torch.Tensor:

        bottleneck_output = self.encoder_activation(
            (self.bottleneck_decoder.forward(input_data=input_data)))

        bottleneck_output = bottleneck_output.reshape(
            (-1, ) + self.before_flatten_dimension)

        reconstructed = self.decoder.forward(input_data=bottleneck_output)

        return reconstructed

    @as_tensor
    def _reconstruction(
            self,
            input_data: Union[torch.Tensor,
                              np.ndarray] = None) -> torch.Tensor:

        reconstructed = self.decoder.forward(input_data=input_data)

        return reconstructed

    def Mu(self,
           input_data: Union[np.ndarray, torch.Tensor] = None,
           to_numpy: bool = False) -> Union[np.ndarray, torch.Tensor]:

        latent = self.projection(input_data=input_data)

        if to_numpy == True:
            return self.z_mean(latent).detach().numpy()
        else:
            return self.z_mean(latent)

    def Sigma(self,
              input_data: Union[np.ndarray, torch.Tensor] = None,
              to_numpy: bool = False) -> Union[np.ndarray, torch.Tensor]:

        latent = self.projection(input_data=input_data)

        if to_numpy == True:
            return torch.exp(self.z_log_var(latent) / 2).detach().numpy()
        else:
            return torch.exp(self.z_log_var(latent) / 2)

    def CoVariance(self,
                   input_data: Union[np.ndarray, torch.Tensor] = None,
                   inv: bool = False,
                   to_numpy: bool = False) -> Union[np.ndarray, torch.Tensor]:

        if inv == False:

            Sigma_inv = 1 / self.Sigma(input_data=input_data)
            covariance = torch.diag_embed(Sigma_inv)

        else:

            Sigma = self.Sigma(input_data=input_data)
            covariance = torch.diag_embed(Sigma)

        if to_numpy == True:
            return covariance.detach().numpy()
        else:
            return covariance

    def latent_gaussian_noisy(
            self,
            input_data: Union[np.ndarray,
                              torch.Tensor] = None) -> torch.Tensor:

        self.mu = self.z_mean(input_data)
        self.log_v = self.z_log_var(input_data)
        eps = self.scale * torch.autograd.Variable(
            torch.randn(*self.log_v.size())).type_as(self.log_v)

        return self.mu + torch.exp(self.log_v / 2.) * eps

    # Evaluating the operation Ũ = D(E(U))
    def reconstruction_forward(
            self,
            input_data: Union[np.ndarray,
                              torch.Tensor] = None) -> torch.Tensor:

        latent = self.projection(input_data=input_data)
        latent_noisy = self.latent_gaussian_noisy(input_data=latent)
        reconstructed = self.reconstruction(input_data=latent_noisy)

        return reconstructed

    # Evaluating the operation Ũ = D(E(U))
    def reconstruction_eval(
            self,
            input_data: Union[np.ndarray,
                              torch.Tensor] = None) -> torch.Tensor:

        encoder_output = self.projection(input_data=input_data)
        latent = self.z_mean(encoder_output)
        reconstructed = self.reconstruction(input_data=latent)

        return reconstructed

    def project(
            self,
            input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype('float32'))

        projected_data_latent = self.Mu(input_data=input_data)

        return projected_data_latent.cpu().detach().numpy()

    def reconstruct(
            self,
            input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype('float32'))

        reconstructed_data = self.reconstruction(input_data=input_data)

        return reconstructed_data.cpu().detach().numpy()

    def eval(self,
             input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype('float32'))

        input_data = input_data.to(self.device)

        return self.reconstruction_eval(
            input_data=input_data).cpu().detach().numpy()
