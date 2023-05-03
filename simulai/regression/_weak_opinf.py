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

import warnings

from typing import List, Union, Tuple, Optional

import os

import numpy as np

from ._opinf import OpInf

MPI_GLOBAL_AVAILABILITY = True

try:
    from mpi4py import MPI
except:
    MPI_GLOBAL_AVAILABILITY = False
    warnings.warn(f"Trying to import MPI in {__file__}.")
    warnings.warn(
        "mpi4py is not installed. If you want to execute MPI jobs, we recommend you install it."
    )

# Operator inference using weak formulation for derivatives
class WeakOpInf(OpInf):
    def __init__(
        self,
        forcing: str = None,
        bias_rescale: float = 1,
        solver: Union[str, callable] = "lstsq",
        parallel: Union[str, None] = None,
        show_log: bool = False,
        engine: str = "numpy",
        test_function : object = None,
    ) -> None:
        """Operator Inference (OpInf) - Weak Formulation

        :param forcing: the kind of forcing to be used, 'linear' or 'nonlinear'
        :type forcing: str
        :param bias_rescale: factor for rescaling the linear coefficients (c_hat)
        :type bias_rescale: float
        :param solver: solver to be used for solving the global system, e. g. 'lstsq'.
        :type solver: Union[str, callable]
        :param parallel: the kind of parallelism to be used (currently, 'mpi' or None)
        :type parallel: str
        :param engine: the engine to be used for constructing the global system (currently just 'numpy')
        :type engine: str
        :param test_function: the test function class used in the inner products
        :tyoe object:
        :return: nothing
        """
        
        super(WeakOpInf, self).__init__(
            forcing = forcing,
            bias_rescale = bias_rescale,
            solver = solver,
            parallel = parallel,
            show_log = show_log,
            engine = engine
        )

        self.test_function = test_function

    def _generate_data_matrices(
        self,
        input_data: np.ndarray = None,
        target_data: np.ndarray = None,
        forcing_data: np.ndarray = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # If forcing_data is None, the Kronecker product is applied just for the field
        # variables, thus reducing to the no forcing term case
        # The field variables quadratic terms are used anyway.

        n_samples = input_data.shape[0]

        quadratic_input_data = self.kronecker_product(a=input_data, b=forcing_data)

        # Matrix used for including constant terms in the operator expression
        unitary_matrix = self.bias_rescale * np.ones((n_samples, 1))

        # Known data matrix (D)
        if forcing_data is not None:
            # Constructing D using purely linear forcing terms
            D = np.hstack(
                [unitary_matrix, input_data, forcing_data, quadratic_input_data]
            )

        else:
            D = np.hstack([unitary_matrix, input_data, quadratic_input_data])

        # Integrating Matrices
        V, Vp, grid = self.test_function.build_V_Vp(t=target_data, x=input_data)
        
        #if len(Vp.shape) > 2:
        #    Res_matrix = np.stack([(Vp[i,...] @ input_data[...,i]) for i in range(input_data.shape[1])]).T
        #else:
            
        Res_matrix = (Vp @ input_data).T
        D = V @ D

        return D, Res_matrix
    
    def construct(
        self,
        input_data: np.ndarray = None,
        target_data: np.ndarray = None,
        forcing_data: np.ndarray = None,
    ) -> None:
        # Collecting information dimensional information from the datasets
        if (
            isinstance(input_data, np.ndarray)
            == isinstance(target_data, np.ndarray)
            == True
        ):
            assert len(input_data.shape) == 2 and len(target_data.shape) == 1, (
                "The input and target data, "
                "must be two-dimensional and one-dimensional, but received shapes"
                f" {input_data.shape} and {target_data.shape}"
            )
            self.n_samples = input_data.shape[0]

            # When there are forcing variables there are extra operators in the model
            if self.forcing is not None:
                assert (
                    forcing_data is not None
                ), "If the forcing terms are used, forcing data must be provided."

                assert len(forcing_data.shape) == 2, (
                    "The forcing data must be two-dimensional,"
                    f" but received shape {forcing_data.shape}"
                )

                assert (
                    input_data.shape[0] == target_data.shape[0] == forcing_data.shape[0]
                ), (
                    "The number of samples is not the same for all the sets with"
                    f"{input_data.shape[0]}, {target_data.shape[0]} and {forcing_data.shape[0]}."
                )

                self.n_forcing_inputs = forcing_data.shape[1]
            # For no forcing cases, the classical form is adopted
            else:
                print("Forcing terms are not being used.")
                assert input_data.shape[0] == target_data.shape[0], (
                    "The number of samples is not the same for all the sets with"
                    f"{input_data.shape[0]} and {target_data.shape[0]}"
                )

            # Number of inputs or degrees of freedom
            self.n_inputs = input_data.shape[1]
            self.n_outputs = input_data.shape[1]

        # When no dataset is provided to fit, it is necessary directly setting up the dimension values
        elif (
            isinstance(input_data, np.ndarray)
            == isinstance(target_data, np.ndarray)
            == False
        ):
            assert self.n_inputs != None and self.n_outputs != None, (
                "It is necessary to provide some" " value to n_inputs and n_outputs"
            )

        else:
            raise Exception(
                "There is no way for executing the system construction"
                " if no dataset or dimension is provided."
            )

        # Defining parameters for the Kronecker product
        if (self.forcing is None) or (self.forcing == "linear"):
            # Getting the upper component indices of a symmetric matrix
            self.i_u, self.j_u = np.triu_indices(self.n_inputs)
            self.n_quadratic_inputs = self.i_u.shape[0]

        # When the forcing interaction is 'nonlinear', there operator H_hat is extended
        elif self.forcing == "nonlinear":
            # Getting the upper component indices of a symmetric matrix
            self.i_u, self.j_u = np.triu_indices(self.n_inputs + self.n_forcing_inputs)
            self.n_quadratic_inputs = self.i_u.shape[0]

        else:
            print(f"The option {self.forcing} is not allowed for the forcing kind.")

        # Number of linear terms
        if forcing_data is not None:
            self.n_forcing_inputs = forcing_data.shape[1]
            self.n_linear_terms = 1 + self.n_inputs + self.n_forcing_inputs
        else:
            self.n_linear_terms = 1 + self.n_inputs

        self.raw_model = False

    