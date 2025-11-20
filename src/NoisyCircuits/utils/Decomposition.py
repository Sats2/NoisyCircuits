from abc import ABC, abstractmethod
import numpy as np


class ShapeMismatchError(Exception):
    """Exception raised for errors in the shape of input arrays."""
    pass

class NonSquareMatrixError(Exception):
    """Exception raised for errors in the shape of input arrays."""
    pass

class NonUnitaryMatrixError(Exception):
    """Exception raised for errors in the shape of input arrays."""
    pass


class Decomposition(ABC):
    """
    Abstract base class for quantum circuit decomposition which defines the interface for various quantum gate operations for different 
    QPUs with varying basis gates.
    """
    def __init__(self,
                 num_qubits:int)->None:
        """
        Initializes the Decomposition class.

        Args:
            num_qubits (int): The number of qubits in the quantum circuit.
        """
        self.num_qubits = num_qubits

    @abstractmethod
    def RZ(self,
           theta:int|float,
           qubit:int):
        r"""
        Applies the RZ gate which is a rotation around the Z-axis by an angle of :math:`\theta`. Effectively applies the unitary:

        .. math::

            RZ(\theta) = \begin{pmatrix}
            e^{-i\theta/2} & 0 \\
            0 & e^{i\theta/2}
            \end{pmatrix}

        Args:
            theta (int | float): The rotation angle.
            qubit (int): The target qubit.
        
        Raises:
            TypeError: If theta is not an integer or float.
            TypeError: If qubit is not an integer.
            ValueError: If qubit is out of range.
        """
        if not isinstance(theta, (int, float)):
            raise TypeError("The angle theta must be an integer or float.")
        if not isinstance(qubit, int):
            raise TypeError("The target qubit must be an integer.")
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"The target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def SX(self,
           qubit:int):
        r"""
        Applies the SX gate which is a square root of the X gate. Effectively applies the unitary:

        .. math::

            SX = \frac{1}{2}\begin{pmatrix}
            1+i & 1-i \\
            1-i & 1+i
            \end{pmatrix}

        Args:
            qubit (int): The target qubit.

        Raises:
            TypeError: If qubit is not an integer.
            ValueError: If qubit is out of range.
        """
        if not isinstance(qubit, int):
            raise TypeError("The target qubit must be an integer.")
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"The target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def X(self,
          qubit:int):
        r"""
        Applies the X gate which is a Pauli-X gate. Effectively applies the unitary:

        .. math::

            X = \begin{pmatrix}
            0 & 1 \\
            1 & 0
            \end{pmatrix}

        Args:
            qubit (int): The target qubit.
        
        Raises:
            TypeError: If qubit is not an integer.
            ValueError: If qubit is out of range.
        """
        if not isinstance(qubit, int):
            raise TypeError("The target qubit must be an integer.")
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"The target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def ECR(self,
            control:int,
            target:int):
        r"""Applies the ECR (Echoed Cross Resonance) gate between two qubits. Effectively applies the unitary:

        .. math::
            ECR(q_0, q_1) = \frac{1}{\sqrt{2}}\begin{pmatrix}
            0 & 0 & 1 & i \\
            0 & 0 & i & 1 \\
            1 & -i & 0 & 0 \\
            -i & 1 & 0 & 0
            \end{pmatrix}

        Args:
            control (int): The control qubit.
            target (int): The target qubit.

        Raises:
            TypeError: If control/target is not an integer.
            ValueError: If control/target is out of range.
        """
        if not isinstance(control, int):
            raise TypeError("The control qubit must be an integer.")
        if not isinstance(target, int):
            raise TypeError("The target qubit must be an integer.")
        if (control < 0 or control >= self.num_qubits) or (target < 0 or target >= self.num_qubits):
            raise ValueError(f"The control or target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def RY(self,
           theta:int|float,
           qubit:int):
        r"""
        Applies the RY gate which is a rotation around the Y-axis by an angle of :math:`\theta`. Effectively applies the unitary:

        .. math::

            RY(\theta) = \begin{pmatrix}
            \cos(\theta/2) & -\sin(\theta/2) \\
            \sin(\theta/2) & \cos(\theta/2)
            \end{pmatrix}

        Args:
            theta (int | float): The rotation angle.
            qubit (int): The target qubit.

        Raises:
            TypeError: If theta is not an integer or float.
            TypeError: If qubit is not an integer.
            ValueError: If the qubit is out of range.
        """
        if not isinstance(theta, (int, float)):
            raise TypeError("The angle theta must be an integer or float.")
        if not isinstance(qubit, int):
            raise TypeError("The target qubit must be an integer.")
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"The target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def RX(self,
           theta:int|float,
           qubit:int):
        r"""
        Applies the RX gate which is a rotation around the X-axis by an angle of :math:`\theta`. Effectively applies the unitary:

        .. math::

            RX(\theta) = \begin{pmatrix}
            \cos(\theta/2) & -i\sin(\theta/2) \\
            -i\sin(\theta/2) & \cos(\theta/2)
            \end{pmatrix}

        Args:
            theta (int | float): The rotation angle.
            qubit (int): The target qubit.

        Raises:
            TypeError: If theta is not an integer or float.
            TypeError: If qubit is not an integer.
            ValueError: If the qubit is out of range.
        """
        if not isinstance(theta, (int, float)):
            raise TypeError("The angle theta must be an integer or float.")
        if not isinstance(qubit, int):
            raise TypeError("The target qubit must be an integer.")
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"The target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def Y(self,
          qubit:int):
        r"""
        Applies the Y gate which is a Pauli-Y gate. Effectively applies the unitary:

        .. math::

            Y = \begin{pmatrix}
            0 & -i \\
            i & 0
            \end{pmatrix}

        Args:
            qubit (int): The target qubit.

        Raises:
            TypeError: If qubit is not an integer.
            ValueError: If the qubit is out of range.
        """
        if not isinstance(qubit, int):
            raise TypeError("The target qubit must be an integer.")
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"The target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def Z(self,
          qubit:int):
        r"""
        Applies the Z gate which is a Pauli-Z gate. Effectively applies the unitary:

        .. math::

            Z = \begin{pmatrix}
            1 & 0 \\
            0 & -1
            \end{pmatrix}

        Args:
            qubit (int): The target qubit.

        Raises:
            TypeError: If qubit is not an integer.
            ValueError: If the qubit is out of range.
        """
        if not isinstance(qubit, int):
            raise TypeError("The target qubit must be an integer.")
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"The target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def H(self,
          qubit:int):
        r"""
        Applies the H gate which is a Hadamard gate. Effectively applies the unitary:

        .. math::

            H = \frac{1}{\sqrt{2}}\begin{pmatrix}
            1 & 1 \\
            1 & -1
            \end{pmatrix}

        Args:
            qubit (int): The target qubit.

        Raises:
            TypeError: If qubit is not an integer.
            ValueError: If the qubit is out of range.
        """
        if not isinstance(qubit, int):
            raise TypeError("The target qubit must be an integer.")
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"The target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def CX(self,
           control:int,
           target:int):
        r"""
        Applies the CX (CNOT) gate between two qubits. Effectively applies the unitary:

        .. math::

            CX(q_0, q_1) = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0
            \end{pmatrix}

        Args:
            control (int): The control qubit.
            target (int): The target qubit.

        Raises:
            TypeError: If control/target is not an integer.
            ValueError: If the control/target is out of range.
        """
        if not isinstance(control, int):
            raise TypeError("The control qubit must be an integer.")
        if not isinstance(target, int):
            raise TypeError("The target qubit must be an integer.")
        if (control < 0 or control >= self.num_qubits) or (target < 0 or target >= self.num_qubits):
            raise ValueError(f"The control or target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def CY(self,
           control:int,
           target:int):
        r"""
        Applies the CY gate between two qubits. Effectively applies the unitary:

        .. math::

            CY(q_0, q_1) = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & -i \\
            0 & 0 & i & 0
            \end{pmatrix}

        Args:
            control (int): The control qubit.
            target (int): The target qubit.

        Raises:
            TypeError: If control/target is not an integer.
            ValueError: If the control/target is out of range.
        """
        if not isinstance(control, int):
            raise TypeError("The control qubit must be an integer.")
        if not isinstance(target, int):
            raise TypeError("The target qubit must be an integer.")
        if (control < 0 or control >= self.num_qubits) or (target < 0 or target >= self.num_qubits):
            raise ValueError(f"The control or target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def CZ(self,
           control:int,
           target:int):
        r"""
        Applies the CZ gate between two qubits. Effectively applies the unitary:

        .. math::

            CZ(q_0, q_1) = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & -1
            \end{pmatrix}

        Args:
            control (int): The control qubit.
            target (int): The target qubit.

        Raises:
            TypeError: If control/target is not an integer.
            ValueError: If the control/target is out of range.
        """
        if not isinstance(control, int):
            raise TypeError("The control qubit must be an integer.")
        if not isinstance(target, int):
            raise TypeError("The target qubit must be an integer.")
        if (control < 0 or control >= self.num_qubits) or (target < 0 or target >= self.num_qubits):
            raise ValueError(f"The control or target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def SWAP(self,
             qubit1:int,
             qubit2:int):
        r"""
        Applies the SWAP gate between two qubits. Effectively applies the unitary:

        .. math::

            SWAP(q_0, q_1) = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1
            \end{pmatrix}

        Args:
            qubit1 (int): The first qubit.
            qubit2 (int): The second qubit.

        Raises:
            TypeError: If qubit1/qubit2 is not an integer.
            ValueError: If qubit1/qubit2 is out of range.
        """
        if not isinstance(qubit1, int):
            raise TypeError("The first qubit must be an integer.")
        if not isinstance(qubit2, int):
            raise TypeError("The second qubit must be an integer.")
        if (qubit1 < 0 or qubit1 >= self.num_qubits) or (qubit2 < 0 or qubit2 >= self.num_qubits):
            raise ValueError(f"The first or second qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def CRX(self,
            theta:int|float,
            control:int,
            target:int):
        r"""
        Applies the CRX (Controlled RX) gate between two qubits. Effectively applies the unitary:

        .. math::

            CRX(\theta, q_0, q_1) = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \cos(\theta/2) & -i\sin(\theta/2) \\
            0 & 0 & -i\sin(\theta/2) & \cos(\theta/2)
            \end{pmatrix}

        Args:
            theta (int | float): The rotation angle.
            control (int): The control qubit.
            target (int): The target qubit.
        
        Raises:
            TypeError: If theta is not an integer or float.
            TypeError: If control/target is not an integer.
            ValueError: If the control/target is out of range.
        """
        if not isinstance(theta, (int, float)):
            raise TypeError("The angle theta must be an integer or float.")
        if not isinstance(control, int):
            raise TypeError("The control qubit must be an integer.")
        if not isinstance(target, int):
            raise TypeError("The target qubit must be an integer.")
        if (control < 0 or control >= self.num_qubits) or (target < 0 or target >= self.num_qubits):
            raise ValueError(f"The control or target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def CRZ(self,
            theta:int|float,
            control:int,
            target:int):
        r"""
        Applies the CRZ (Controlled RZ) gate between two qubits. Effectively applies the unitary:

        .. math::
            CRZ(\theta, q_0, q_1) = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \exp(-i\theta/2) & 0 \\
            0 & 0 & 0 & \exp(i\theta/2)
            \end{pmatrix}

        Args:
            theta (int | float): The rotation angle.
            control (int): The control qubit.
            target (int): The target qubit.
        
        Raises:
            TypeError: If theta is not an integer or float.
            TypeError: If control/target is not an integer.
            ValueError: If the control/target is out of range.
        """
        if not isinstance(theta, (int, float)):
            raise TypeError("The angle theta must be an integer or float.")
        if not isinstance(control, int):
            raise TypeError("The control qubit must be an integer.")
        if not isinstance(target, int):
            raise TypeError("The target qubit must be an integer.")
        if (control < 0 or control >= self.num_qubits) or (target < 0 or target >= self.num_qubits):
            raise ValueError(f"The control or target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def CRY(self,
            theta:int|float,
            control:int,
            target:int):
        r"""
        Applies the CRY (Controlled RY) gate between two qubits. Effectively applies the unitary:

        .. math::
            CRY(\theta, q_0, q_1) = \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \cos(\theta/2) & \sin(\theta/2) \\
            0 & 0 & \sin(\theta/2) & \cos(\theta/2)
            \end{pmatrix}

        Args:
            theta (int | float): The rotation angle.
            control (int): The control qubit.
            target (int): The target qubit.
        
        Raises:
            TypeError: If theta is not an integer or float.
            TypeError: If control/target is not an integer.
            ValueError: If the control/target is out of range.
        """
        if not isinstance(theta, (int, float)):
            raise TypeError("The angle theta must be an integer or float.")
        if not isinstance(control, int):
            raise TypeError("The control qubit must be an integer.")
        if not isinstance(target, int):
            raise TypeError("The target qubit must be an integer.")#
        if (control < 0 or control >= self.num_qubits) or (target < 0 or target >= self.num_qubits):
            raise ValueError(f"The control or target qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def RZZ(self,
            theta:int|float,
            qubit1:int,
            qubit2:int):
        r"""
        Applies the RZZ Coupling Gate (Ising ZZ) :math:`ZZ(\theta)` which effectively applies the unitary:

        .. math::
            ZZ(\theta) = \begin{pmatrix}
            \exp(-i\theta/2) & 0 & 0 & 0 \\
            0 & \exp(i\theta/2) & 0 & 0 \\
            0 & 0 & \exp(i\theta/2) & 0 \\
            0 & 0 & 0 & \exp(-i\theta/2)
            \end{pmatrix}

        Args:
            theta (int | float): The angle of rotation.
            qubit1 (int): The first qubit.
            qubit2 (int): The second qubit.
        
        Raises:
            TypeError: If theta is not an integer or float.
            TypeError: If qubit1/qubit2 is not an integer.
            ValueError: If qubit1/qubit2 is out of range.
        """
        if not isinstance(theta, (int, float)):
            raise TypeError("The angle theta must be an integer or float.")
        if not isinstance(qubit1, int):
            raise TypeError("The first qubit must be an integer.")
        if not isinstance(qubit2, int):
            raise TypeError("The second qubit must be an integer.")
        if (qubit1 < 0 or qubit1 >= self.num_qubits) or (qubit2 < 0 or qubit2 >= self.num_qubits):
            raise ValueError(f"The first or second qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def RXX(self,
            theta:int|float,
            qubit1:int,
            qubit2:int):
        r"""
        Applies the RXX Coupling Gate (Ising XX) :math:`XX(\theta)` which effectively applies the unitary:

        .. math::
            XX(\theta) = \begin{pmatrix}
            \cos(\theta/2) & 0 & 0 & -i\sin(\theta/2) \\
            0 & \cos(\theta/2) & -i\sin(\theta/2) & 0 \\
            0 & -i\sin(\theta/2) & \cos(\theta/2) & 0 \\
            -i\sin(\theta/2) & 0 & 0 & \cos(\theta/2)
            \end{pmatrix}

        Args:
            theta (int | float): The angle of rotation.
            qubit1 (int): The first qubit.
            qubit2 (int): The second qubit.
        
        Raises:
            TypeError: If theta is not an integer or float.
            TypeError: If qubit1/qubit2 is not an integer.
            ValueError: If qubit1/qubit2 is out of range.
        """
        if not isinstance(theta, (int, float)):
            raise TypeError("The angle theta must be an integer or float.")
        if not isinstance(qubit1, int):
            raise TypeError("The first qubit must be an integer.")
        if not isinstance(qubit2, int):
            raise TypeError("The second qubit must be an integer.")
        if (qubit1 < 0 or qubit1 >= self.num_qubits) or (qubit2 < 0 or qubit2 >= self.num_qubits):
            raise ValueError(f"The first or second qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def RYY(self,
            theta:int|float,
            qubit1:int,
            qubit2:int):
        r"""
        Applies the RYY Coupling Gate (Ising YY) :math:`YY(\theta)` which effectively applies the unitary:

        .. math::
            YY(\theta) = \begin{pmatrix}
            \cos(\theta/2) & 0 & 0 & i\sin(\theta/2) \\
            0 & \cos(\theta/2) & -i\sin(\theta/2) & 0 \\
            0 & -i\sin(\theta/2) & \cos(\theta/2) & 0 \\
            i\sin(\theta/2) & 0 & 0 & \cos(\theta/2)
            \end{pmatrix}

        Args:
            theta (int | float): The angle of rotation.
            qubit1 (int): The first qubit.
            qubit2 (int): The second qubit.
        
        Raises:
            TypeError: If theta is not an integer or float.
            TypeError: If qubit1/qubit2 is not an integer.
            ValueError: If qubit1/qubit2 is out of range.
        """
        if not isinstance(theta, (int, float)):
            raise TypeError("The angle theta must be an integer or float.")
        if not isinstance(qubit1, int):
            raise TypeError("The first qubit must be an integer.")
        if not isinstance(qubit2, int):
            raise TypeError("The second qubit must be an integer.")
        if (qubit1 < 0 or qubit1 >= self.num_qubits) or (qubit2 < 0 or qubit2 >= self.num_qubits):
            raise ValueError(f"The first or second qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def apply_swap_decomposition(self,
              qubit1:int,
              qubit2:int):
        """
        Applies the SWAP decomposition to the circuit for every SWAP call.

        Args:
            qubit1 (int): The first qubit.
            qubit2 (int): The second qubit.
        
        Raises:
            TypeError: If qubit1/qubit2 is not an integer.
            ValueError: If qubit1/qubit2 is out of range.
        """
        if not isinstance(qubit1, int):
            raise TypeError("The first qubit must be an integer.")
        if not isinstance(qubit2, int):
            raise TypeError("The second qubit must be an integer.")
        if (qubit1 < 0 or qubit1 >= self.num_qubits) or (qubit2 < 0 or qubit2 >= self.num_qubits):
            raise ValueError(f"The first or second qubit is out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True

    @abstractmethod
    def apply_unitary(self,
                      unitary_matrix:np.ndarray,
                      qubits:list[int]):
        """
        Applies a unitary operation to the specified qubits.

        Args:
            unitary_matrix (np.ndarray): The unitary matrix to apply.
            qubits (list[int]): The list of qubits to which the unitary matrix will be applied.
        
        Raises:
            NonSquareMatrixError: If the unitary matrix is not square.
            ShapeMismatchError: If the shape of the unitary matrix does not match the state of the qubits for the provided number of qubits.
            NonUnitaryMatrixError: If the matrix is not unitary. 
            TypeError: If any qubit in the qubits list is not an integer.
            ValueError: If any qubit in the qubits list is out of range.
        """
        if not unitary_matrix.shape[0] == unitary_matrix.shape[1]:
            raise NonSquareMatrixError("The provided matrix is not square.")
        if not unitary_matrix.shape[0] == 2**len(qubits):
            raise ShapeMismatchError("The shape of the unitary matrix does not match the state of the qubits.")
        if not np.allclose(np.eye(unitary_matrix.shape[0]), unitary_matrix.conj().T @ unitary_matrix):
            raise NonUnitaryMatrixError("The provided matrix is not unitary.")
        if not all(isinstance(qubit, int) for qubit in qubits):
            raise TypeError("All qubits must be integers.")
        if any(qubit < 0 or qubit >= self.num_qubits for qubit in qubits):
            raise ValueError(f"One or more qubits are out of range. The valid range is from 0 to {self.num_qubits - 1}.")
        return True