import socket
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GoSANSchedulerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="127.0.0.1", port=1337, num_disks=4):
        super().__init__()

        self.host = host
        self.port = port
        self.num_disks = num_disks

        self.sock = None
        self.file = None

        self.state_dim = num_disks * 3  # queues + service rates + alive mask
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_disks)

    # ---- TCP helpers ----

    def _connect(self):
        self.sock = socket.create_connection((self.host, self.port))
        self.file = self.sock.makefile("rwb")

    def _send_action(self, action):
        # be robust to different action types from SB3 / eval
        if isinstance(action, np.ndarray):
            action = int(action.flatten()[0])
        else:
            action = int(action)
        msg = json.dumps({"action": action}).encode() + b"\n"
        self.file.write(msg)
        self.file.flush()

    def _recv_response(self):
        line = self.file.readline()
        if not line:
            raise RuntimeError("connection closed by Go server")

        data = json.loads(line)
        state = np.array(data.get("state", []), dtype=np.float32)
        reward = float(data.get("reward", 0.0))
        terminated = bool(data.get("terminated", False))
        metrics = data.get("metrics", {})
        return state, reward, terminated, metrics

    # ---- Gym API ----

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # fresh connection per episode (server = one episode per TCP conn)
        self.close()
        self._connect()

        state, _, terminated, _ = self._recv_response()
        if terminated:
            # extremely unlikely on first message, but handle gracefully
            return state, {}
        return state, {}

    def step(self, action):
        if self.sock is None or self.file is None:
            raise RuntimeError("Environment step called with no active connection.")

        self._send_action(action)
        state, reward, terminated, metrics = self._recv_response()

        info = metrics  # contains last_latency, step, etc.
        truncated = False

        if terminated:
            # Episode ended on Go side; we keep socket cleanup for reset/close.
            pass

        return state, reward, terminated, truncated, info

    def close(self):
        try:
            if self.file is not None:
                self.file.close()
            if self.sock is not None:
                self.sock.close()
        finally:
            self.file = None
            self.sock = None
