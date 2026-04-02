"""Stub for DL/DPL model weight serialization.

This module provides a placeholder for DL weight exchange.
Replace dummy_weights() with actual model weight extraction
when integrating with a real DPL training stack (e.g., PyTorch).
"""

import random


class DLPayload:
    """Handles serialization/deserialization of DL model weights for V2V exchange."""

    @staticmethod
    def dummy_weights(num_layers=3, layer_size=10):
        """Generate dummy model weights for testing.

        Returns:
            dict with layer names as keys and weight lists as values.
        """
        weights = {}
        for i in range(num_layers):
            weights[f"layer_{i}"] = [
                round(random.gauss(0, 0.1), 4) for _ in range(layer_size)
            ]
        return weights

    @staticmethod
    def serialize_weights(weights):
        """Serialize model weights to a dict suitable for V2V message payload.

        Args:
            weights: dict of {layer_name: list_of_floats}

        Returns:
            dict payload ready for V2VMessage.
        """
        return {
            "type": "dl_model_update",
            "round": 0,
            "weights": weights,
        }

    @staticmethod
    def deserialize_weights(payload):
        """Extract model weights from a V2V message payload.

        Args:
            payload: dict from V2VMessage.payload

        Returns:
            dict of {layer_name: list_of_floats} or None.
        """
        if payload.get("type") in {"dl_model_update", "fl_model_update"}:
            return payload.get("weights")
        return None


# Backward-compatible alias for older imports.
FLPayload = DLPayload
