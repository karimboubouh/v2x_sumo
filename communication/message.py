"""V2V message data structures."""

import uuid
from dataclasses import dataclass, field


@dataclass
class V2VMessage:
    """A message exchanged between two vehicles."""
    sender_id: str
    receiver_id: str
    timestamp: float          # simulation time when created
    msg_type: str             # "hello", "data", "fl_weights"
    payload: dict = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    delivered: bool = False
    delivery_time: float = 0.0
    attempts: int = 0

    def __str__(self):
        status = "OK" if self.delivered else "pending"
        return (
            f"[{self.timestamp:.0f}s] {self.sender_id} -> {self.receiver_id}: "
            f"{self.msg_type} ({status})"
        )
