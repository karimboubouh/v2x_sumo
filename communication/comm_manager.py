"""Communication manager: neighbor discovery, message queuing, and delivery."""

import random
from collections import defaultdict

import config
from communication.v2v_link import V2VLink, compute_link
from communication.message import V2VMessage


class CommManager:
    """Manages V2V communication between vehicles."""

    def __init__(
        self,
        comm_range=None,
        max_neighbors=None,
        power_dbm=None,
        path_loss_exp=None,
        noise_floor_dbm=None,
        snr_threshold_db=None,
        beacon_interval=None,
        message_ttl=None,
    ):
        self.comm_range = comm_range or config.COMM_RANGE
        self.max_neighbors = max_neighbors if max_neighbors is not None else config.MAX_NEIGHBORS
        self.power_dbm = power_dbm or config.COMM_POWER_DBM
        self.path_loss_exp = path_loss_exp or config.PATH_LOSS_EXPONENT
        self.noise_floor_dbm = noise_floor_dbm or config.NOISE_FLOOR_DBM
        self.snr_threshold_db = snr_threshold_db or config.SNR_THRESHOLD_DB
        self.beacon_interval = beacon_interval or config.BEACON_INTERVAL
        self.message_ttl = message_ttl or config.MESSAGE_TTL

        self._active_links = []       # Current V2V links
        self._outgoing_queue = []     # Messages waiting for delivery
        self._last_beacon = {}        # {vehicle_id: last_beacon_time}
        self._neighbors = defaultdict(list)  # {vehicle_id: [neighbor_ids]}
        self._stats = {"sent": 0, "delivered": 0, "dropped": 0}

    def update(self, vehicle_states, sim_time):
        """Update communication state for the current simulation step.

        Args:
            vehicle_states: dict of {vehicle_id: VehicleState}
            sim_time: current simulation time in seconds

        Returns:
            list of V2VMessage that were delivered or generated this step
        """
        # 1. Compute all pairwise links
        self._compute_links(vehicle_states)

        # 2. Generate hello beacons
        beacons = self._generate_beacons(vehicle_states, sim_time)

        # 3. Attempt to deliver queued messages
        delivered = self._deliver_messages(sim_time)

        return beacons + delivered

    def send_message(self, sender_id, receiver_id, msg_type, payload, sim_time):
        """Queue a message for delivery."""
        msg = V2VMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            timestamp=sim_time,
            msg_type=msg_type,
            payload=payload,
        )
        self._outgoing_queue.append(msg)
        self._stats["sent"] += 1
        return msg

    def get_active_links(self):
        """Return current active V2V links for rendering."""
        return self._active_links

    def get_neighbors(self, vehicle_id):
        """Return list of neighbor vehicle IDs within communication range."""
        return self._neighbors.get(vehicle_id, [])

    def get_stats(self):
        """Return communication statistics."""
        return dict(self._stats)

    def _compute_links(self, vehicle_states):
        """Compute pairwise links between all vehicles, capped at max_neighbors each."""
        self._active_links = []
        self._neighbors = defaultdict(list)

        veh_ids = list(vehicle_states.keys())

        # Pass 1: compute all candidate links within COMM_RANGE
        all_links = []
        per_vehicle = defaultdict(list)
        for i in range(len(veh_ids)):
            for j in range(i + 1, len(veh_ids)):
                id_a = veh_ids[i]
                id_b = veh_ids[j]
                state_a = vehicle_states[id_a]
                state_b = vehicle_states[id_b]

                link = compute_link(
                    sender_id=id_a,
                    receiver_id=id_b,
                    pos_a=(state_a.x, state_a.y),
                    pos_b=(state_b.x, state_b.y),
                    power_dbm=self.power_dbm,
                    path_loss_exp=self.path_loss_exp,
                    noise_floor_dbm=self.noise_floor_dbm,
                    snr_threshold_db=self.snr_threshold_db,
                    comm_range=self.comm_range,
                )

                if link and link.is_active:
                    all_links.append(link)
                    per_vehicle[id_a].append(link)
                    per_vehicle[id_b].append(link)

        # Pass 2: each vehicle keeps its top-max_neighbors links by quality
        accepted = set()
        for links in per_vehicle.values():
            top = sorted(links, key=lambda lnk: lnk.quality, reverse=True)[:self.max_neighbors]
            for lnk in top:
                accepted.add(frozenset({lnk.sender_id, lnk.receiver_id}))

        # Build final active_links and neighbors from the accepted set (union semantics)
        for link in all_links:
            if frozenset({link.sender_id, link.receiver_id}) in accepted:
                self._active_links.append(link)
                self._neighbors[link.sender_id].append(link.receiver_id)
                self._neighbors[link.receiver_id].append(link.sender_id)

    def _generate_beacons(self, vehicle_states, sim_time):
        """Generate periodic hello beacons from each vehicle."""
        beacons = []
        for veh_id in vehicle_states:
            last = self._last_beacon.get(veh_id, -self.beacon_interval)
            if sim_time - last >= self.beacon_interval:
                self._last_beacon[veh_id] = sim_time
                neighbors = self._neighbors.get(veh_id, [])
                for neighbor_id in neighbors:
                    msg = V2VMessage(
                        sender_id=veh_id,
                        receiver_id=neighbor_id,
                        timestamp=sim_time,
                        msg_type="hello",
                        payload={"speed": vehicle_states[veh_id].speed},
                        delivered=True,
                        delivery_time=sim_time,
                    )
                    beacons.append(msg)
                    self._stats["sent"] += 1
                    self._stats["delivered"] += 1
        return beacons

    def _deliver_messages(self, sim_time):
        """Attempt to deliver queued messages based on link quality."""
        delivered = []
        remaining = []

        # Build a quick lookup for active links
        link_map = {}
        for link in self._active_links:
            link_map[(link.sender_id, link.receiver_id)] = link
            link_map[(link.receiver_id, link.sender_id)] = link

        for msg in self._outgoing_queue:
            if msg.delivered:
                continue

            msg.attempts += 1
            key = (msg.sender_id, msg.receiver_id)
            link = link_map.get(key)

            if link and random.random() < link.quality:
                # Successful delivery
                msg.delivered = True
                msg.delivery_time = sim_time
                delivered.append(msg)
                self._stats["delivered"] += 1
            elif msg.attempts >= self.message_ttl:
                # TTL expired
                self._stats["dropped"] += 1
            else:
                # Keep in queue for retry
                remaining.append(msg)

        self._outgoing_queue = remaining
        return delivered

    def _find_link(self, sender_id, receiver_id):
        """Find the link between two vehicles."""
        for link in self._active_links:
            if (link.sender_id == sender_id and link.receiver_id == receiver_id) or \
               (link.sender_id == receiver_id and link.receiver_id == sender_id):
                return link
        return None
