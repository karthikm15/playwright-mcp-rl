"""State encoder for converting snapshots to vectors."""

import torch
import torch.nn as nn
from typing import Dict, Any, List


class StateEncoder(nn.Module):
    """Encodes browser snapshot to fixed-size vector."""
    
    def __init__(self, max_elements: int = 50, element_dim: int = 64):
        super().__init__()
        """
        Initialize state encoder.
        
        Args:
            max_elements: Maximum number of elements to consider
            element_dim: Dimension for each element encoding
        """
        self.max_elements = max_elements
        self.element_dim = element_dim
        
        # Element type embeddings
        self.type_to_idx = {
            'textbox': 0,
            'button': 1,
            'radio': 2,
            'checkbox': 3,
            'link': 4,
        }
        self.num_types = len(self.type_to_idx)
        
        # Embedding for element type
        self.type_embedding = nn.Embedding(self.num_types, 16)
        
        # Element encoder: type_embed + name_embed + value_embed
        # name and value will be hashed to indices
        self.name_embedding = nn.Embedding(1000, 32)  # Hash name to 0-999
        self.value_embedding = nn.Embedding(1000, 16)  # Hash value to 0-999
        
        # Final element encoding dimension
        self.element_encoding_dim = 16 + 32 + 16  # type + name + value
    
    def encode_element(self, element: Dict[str, Any]) -> torch.Tensor:
        """Encode a single element to vector."""
        # Type embedding
        elem_type = element.get('type', '').lower()
        type_idx = self.type_to_idx.get(elem_type, 0)
        type_emb = self.type_embedding(torch.tensor(type_idx, dtype=torch.long))
        
        # Name embedding (hash name to index)
        name = element.get('name', '')
        name_hash = hash(name) % 1000
        name_emb = self.name_embedding(torch.tensor(abs(name_hash), dtype=torch.long))
        
        # Value embedding (hash value to index)
        value = str(element.get('value', ''))
        value_hash = hash(value) % 1000
        value_emb = self.value_embedding(torch.tensor(abs(value_hash), dtype=torch.long))
        
        return torch.cat([type_emb, name_emb, value_emb])
    
    def encode_snapshot(self, snapshot: Dict[str, Any]) -> torch.Tensor:
        """
        Encode snapshot to fixed-size vector.
        
        Args:
            snapshot: Snapshot dict with 'elements' list
        
        Returns:
            state_vector: [state_dim] tensor
        """
        elements = snapshot.get('elements', [])
        
        # Encode each element
        element_encodings = []
        for elem in elements[:self.max_elements]:
            elem_enc = self.encode_element(elem)
            element_encodings.append(elem_enc)
        
        # Pad or truncate to max_elements
        while len(element_encodings) < self.max_elements:
            element_encodings.append(torch.zeros(self.element_encoding_dim))
        
        element_encodings = element_encodings[:self.max_elements]
        
        # Stack and flatten
        elements_tensor = torch.stack(element_encodings)  # [max_elements, element_dim]
        state_vector = elements_tensor.flatten()  # [max_elements * element_dim]
        
        return state_vector
    
    def get_state_dim(self) -> int:
        """Get dimension of encoded state."""
        return self.max_elements * self.element_encoding_dim

