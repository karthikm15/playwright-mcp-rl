"""Transformer-based state encoder for converting snapshots to vectors."""

import torch
import torch.nn as nn
import math
from typing import Dict, Any


class TransformerStateEncoder(nn.Module):
    """Encodes browser snapshot to fixed-size vector using transformer."""
    
    def __init__(self, max_elements: int = 50, element_dim: int = 64, 
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        """
        Initialize transformer state encoder.
        
        Args:
            max_elements: Maximum number of elements to consider
            element_dim: Output dimension per element (used for compatibility, 
                        actual output is d_model)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
        """
        super().__init__()
        self.max_elements = max_elements
        self.d_model = d_model
        
        # Element type embeddings (same as StateEncoder)
        self.type_to_idx = {
            'textbox': 0,
            'button': 1,
            'radio': 2,
            'checkbox': 3,
            'link': 4,
        }
        self.num_types = len(self.type_to_idx)
        
        # Element embeddings
        self.type_embedding = nn.Embedding(self.num_types, 16)
        self.name_embedding = nn.Embedding(1000, 32)  # Hash name to 0-999
        self.value_embedding = nn.Embedding(1000, 16)  # Hash value to 0-999
        
        # State features: filled/unfilled status
        # - For textbox: 1 if value is non-empty, 0 otherwise
        # - For radio/checkbox: 1 if checked, 0 otherwise
        # - For other elements: always 0
        # This helps detect small state changes
        self.state_feature_dim = 3  # [is_filled, is_checked, has_value]
        
        # Project element features to d_model
        element_feat_dim = 16 + 32 + 16 + self.state_feature_dim  # type + name + value + state_features
        self.element_projection = nn.Linear(element_feat_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_elements)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global completion features: track form completion status
        # This helps the policy know when all required fields are filled
        self.completion_feature_dim = 3  # [num_filled_textboxes, num_checked_radios, completion_ratio]
        
        # Project completion features and pooled representation
        # Output dimension is d_model + completion features
        self.completion_projection = nn.Linear(self.completion_feature_dim, 16)
        self.output_projection = nn.Linear(d_model + 16, d_model)
        self.output_dim = d_model
    
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
        
        # State features: capture filled/unfilled status
        # This is critical for detecting small state changes
        elem_type_lower = elem_type
        value_str = str(element.get('value', ''))
        checked = element.get('checked', False)
        
        # is_filled: 1 if element has been filled (textbox with text, or checked radio/checkbox)
        is_filled = 0.0
        if elem_type_lower == 'textbox' and value_str.strip():
            is_filled = 1.0
        elif elem_type_lower in ['radio', 'checkbox'] and checked:
            is_filled = 1.0
        
        # is_checked: 1 if radio/checkbox is checked
        is_checked = 1.0 if (elem_type_lower in ['radio', 'checkbox'] and checked) else 0.0
        
        # has_value: 1 if element has any value (for textboxes)
        has_value = 1.0 if (elem_type_lower == 'textbox' and value_str.strip()) else 0.0
        
        state_features = torch.tensor([is_filled, is_checked, has_value], dtype=torch.float32)
        
        # Concatenate features
        element_features = torch.cat([type_emb, name_emb, value_emb, state_features])
        
        # Project to d_model
        element_vec = self.element_projection(element_features.unsqueeze(0))
        
        return element_vec.squeeze(0)
    
    def encode_snapshot(self, snapshot: Dict[str, Any]) -> torch.Tensor:
        """
        Encode snapshot to fixed-size vector using transformer.
        
        Args:
            snapshot: Snapshot dict with 'elements' list
        
        Returns:
            state_vector: [d_model] tensor
        """
        elements = snapshot.get('elements', [])
        
        # Encode each element
        element_encodings = []
        for elem in elements[:self.max_elements]:
            elem_enc = self.encode_element(elem)
            element_encodings.append(elem_enc)
        
        # Pad to max_elements
        while len(element_encodings) < self.max_elements:
            element_encodings.append(torch.zeros(self.d_model))
        
        element_encodings = element_encodings[:self.max_elements]
        
        # Stack: [max_elements, d_model]
        elements_tensor = torch.stack(element_encodings)
        
        # Add batch dimension: [1, max_elements, d_model]
        elements_tensor = elements_tensor.unsqueeze(0)
        
        # Add positional encoding
        elements_tensor = self.pos_encoding(elements_tensor)
        
        # Transformer encoder: [1, max_elements, d_model]
        encoded = self.transformer(elements_tensor)
        
        # Mean pool over sequence length: [1, d_model]
        pooled = encoded.mean(dim=1).squeeze(0)  # [d_model]
        
        # Compute global completion features
        # Count filled textboxes, checked radios, and overall completion ratio
        num_textboxes = 0
        num_filled_textboxes = 0
        num_radios = 0
        num_checked_radios = 0
        
        for elem in elements[:self.max_elements]:
            elem_type = elem.get('type', '').lower()
            value_str = str(elem.get('value', ''))
            checked = elem.get('checked', False)
            
            if elem_type == 'textbox':
                num_textboxes += 1
                if value_str.strip():
                    num_filled_textboxes += 1
            elif elem_type == 'radio':
                num_radios += 1
                if checked:
                    num_checked_radios += 1
        
        # Completion ratio: how many required fields are filled
        total_required = num_textboxes + num_radios
        total_filled = num_filled_textboxes + num_checked_radios
        completion_ratio = total_filled / total_required if total_required > 0 else 0.0
        
        # Normalize counts (divide by max_elements to keep in [0, 1])
        completion_features = torch.tensor([
            num_filled_textboxes / max(self.max_elements, 1),
            num_checked_radios / max(self.max_elements, 1),
            completion_ratio
        ], dtype=torch.float32)
        
        # Project completion features
        completion_emb = self.completion_projection(completion_features)
        
        # Concatenate pooled representation with completion features
        combined = torch.cat([pooled, completion_emb])
        
        # Final projection to output dimension
        state_vector = self.output_projection(combined)
        
        return state_vector
    
    def get_state_dim(self) -> int:
        """Get dimension of encoded state."""
        return self.output_dim


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

