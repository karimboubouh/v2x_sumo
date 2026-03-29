"""V2V link quality model based on log-distance path loss."""

import math
from dataclasses import dataclass


@dataclass
class V2VLink:
    """Represents a communication link between two vehicles."""
    sender_id: str
    receiver_id: str
    distance: float       # meters
    rssi_dbm: float       # received signal strength indicator
    snr_db: float         # signal-to-noise ratio
    quality: float        # 0.0 to 1.0 (delivery probability)
    is_active: bool       # whether link quality is above minimum threshold


def compute_link(
    sender_id,
    receiver_id,
    pos_a,
    pos_b,
    power_dbm=-30.0,
    path_loss_exp=2.5,
    noise_floor_dbm=-90.0,
    snr_threshold_db=10.0,
    comm_range=200.0,
):
    """Compute link quality between two positions using log-distance path loss.

    Args:
        sender_id: ID of the sending vehicle.
        receiver_id: ID of the receiving vehicle.
        pos_a: (x, y) position of sender in meters.
        pos_b: (x, y) position of receiver in meters.
        power_dbm: Transmit power in dBm (reference at 1m).
        path_loss_exp: Path loss exponent (2.0=free space, 2.5=urban).
        noise_floor_dbm: Receiver noise floor in dBm.
        snr_threshold_db: SNR threshold for 50% delivery.
        comm_range: Maximum communication range in meters.

    Returns:
        V2VLink with computed metrics, or None if out of range.
    """
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    distance = math.sqrt(dx * dx + dy * dy)

    # Out of range
    if distance > comm_range:
        return None

    # Avoid log(0)
    if distance < 1.0:
        distance = 1.0

    # Log-distance path loss: RSSI = P_tx - 10 * n * log10(d)
    rssi_dbm = power_dbm - 10.0 * path_loss_exp * math.log10(distance)

    # Signal-to-noise ratio
    snr_db = rssi_dbm - noise_floor_dbm

    # Map SNR to delivery probability (quality)
    if snr_db >= snr_threshold_db:
        # Above threshold: high quality, asymptotically approaching 1.0
        quality = min(1.0, 0.5 + (snr_db - snr_threshold_db) / 20.0)
    elif snr_db > 0:
        # Below threshold but positive SNR: degraded quality
        quality = max(0.0, snr_db / snr_threshold_db * 0.5)
    else:
        quality = 0.0

    is_active = quality > 0.05  # Minimum usable quality

    return V2VLink(
        sender_id=sender_id,
        receiver_id=receiver_id,
        distance=distance,
        rssi_dbm=rssi_dbm,
        snr_db=snr_db,
        quality=quality,
        is_active=is_active,
    )
