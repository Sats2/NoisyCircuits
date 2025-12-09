"""
This module builds the connectivity map for the given hardware for the number of qubits used. This module also generates the shortest path to connect qubits in the hardware to ensure that the least possible number of SWAP operations are performed to connect two qubits on the system hardware. It is not meant to be used as a standalone module, but could be used to determine qubit connection paths for a given qubit layout which is provided as a dictionary.
"""
from collections import deque


class QubitCouplingMap:
    """
    Module that builds the connectivity map for the given hardware for the number of qubits used. This module also generates the shortest path to connect qubits in the hardware to ensure that the least possible number of SWAP operations are performed to connect two qubits on the system hardware.
    """
    def __init__(self,
                 num_qubits:int,
                 connectivity:dict)->None:
        """
        Constructor for the module.

        Args:
            num_qubits (int): Total number of qubits in the system that are utilized.
            connectivity (dict): Dictionary containing the directionality of the qubits (control-target pairings.)
        """
        self.num_qubits = num_qubits
        self.connectivity = connectivity
        self.logical_to_physical = {i: i for i in range(num_qubits)}
        self.physical_to_logical = {i: i for i in range(num_qubits)}

    def find_shortest_path(self, 
                           start:int, 
                           end:int)->list:
        """
        Find the shortest path between two qubits using BFS.
        
        Args:
            start (int): Starting qubit
            end (int): Target qubit
            
        Returns:
            list: Path from start to end qubit
        """
        if start == end:
            return [start]
            
        # Use connectivity map directly as it's already in adjacency list format
        # Ensure all qubits are represented in the graph
        graph = {i: [] for i in range(self.num_qubits)}
        for qubit, neighbors in self.connectivity.items():
            graph[qubit] = neighbors
        
        # BFS to find shortest path
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in graph[current]:
                if neighbor == end:
                    return path + [neighbor]
                    
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        raise ValueError(f"No path found between qubits {start} and {end}")

    def generate_swap_sequence(self, 
                               logical_control:int, 
                               logical_target:int)->tuple:
        """
        Generate SWAP sequence to bring qubits close enough for interaction.
        
        Args:
            logical_control (int): Logical control qubit
            logical_target (int): Logical target qubit
            
        Returns:
            tuple: (forward_swaps, reverse_swaps, final_control_pos, final_target_pos)
        """
        # Get current physical positions
        control_pos = self.logical_to_physical[logical_control]
        target_pos = self.logical_to_physical[logical_target]
        
        # Check if they're already connected
        if target_pos in self.connectivity.get(control_pos, []) or control_pos in self.connectivity.get(target_pos, []):
            return [], [], control_pos, target_pos
        
        # Find shortest path between control and target
        path = self.find_shortest_path(control_pos, target_pos)
        
        # Generate swaps to move target qubit towards control
        forward_swaps = []
        current_target_pos = target_pos
        
        # Move target qubit step by step towards control
        for i in range(len(path) - 2, 0, -1):  # Move from target towards control
            swap_pos = path[i]
            if current_target_pos != swap_pos:
                forward_swaps.append((current_target_pos, swap_pos))
                # Update qubit mappings
                self.update_mapping_after_swap(current_target_pos, swap_pos)
                current_target_pos = swap_pos
        
        final_control_pos = self.logical_to_physical[logical_control]
        final_target_pos = self.logical_to_physical[logical_target]
        
        # Generate reverse swaps (in reverse order)
        reverse_swaps = forward_swaps[::-1]

        # Restore original mapping after generating swaps
        self.reset_qubit_mapping()
        
        return forward_swaps, reverse_swaps, final_control_pos, final_target_pos
    
    def update_mapping_after_swap(self, 
                                  qubit1:int, 
                                  qubit2:int):
        """
        Update logical-physical mapping after a SWAP operation.
        
        Args:
            qubit1 (int): First qubit in SWAP
            qubit2 (int): Second qubit in SWAP
        """
        # Get logical qubits at these physical positions
        logical1 = self.physical_to_logical[qubit1]
        logical2 = self.physical_to_logical[qubit2]
        
        # Swap the mappings
        self.logical_to_physical[logical1] = qubit2
        self.logical_to_physical[logical2] = qubit1
        self.physical_to_logical[qubit1] = logical2
        self.physical_to_logical[qubit2] = logical1

    def get_qubit_mapping(self) -> dict:
        """
        Returns the current logical to physical qubit mapping.
        
        Returns:
            dict: Mapping from logical to physical qubits.
        """
        return self.logical_to_physical.copy()

    def reset_qubit_mapping(self):
        """
        Resets the qubit mapping to the initial state.
        """
        self.logical_to_physical = {i: i for i in range(self.num_qubits)}
        self.physical_to_logical = {i: i for i in range(self.num_qubits)}