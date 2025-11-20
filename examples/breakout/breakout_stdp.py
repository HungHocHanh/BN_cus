import os   # <<< MỚI
import torch  # <<< MỚI

from bindsnet.encoding import bernoulli
from bindsnet.environment import GymEnvironment
from bindsnet.learning import MSTDP
from bindsnet.network import Network
from bindsnet.network.nodes import Input, IzhikevichNodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_softmax

CHECKPOINT_PATH = "breakout_checkpoint.pth"  # <<< MỚI
CHECKPOINT_INTERVAL = 100  # lưu mỗi 100 episode  <<< MỚI

# Build network.
network = Network(dt=1.0)

# Layers of neurons.
inpt = Input(n=80 * 80, shape=[1, 1, 1, 80, 80], traces=True)
middle = IzhikevichNodes(n=100, traces=True)
out = IzhikevichNodes(n=4, rfrace=0, traces=True)

# Connections between layers.
inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
middle_out = Connection(
    source=middle,
    target=out,
    wmin=0,
    wmax=1,
    update_rule=MSTDP,
    nu=1e-1,
    norm=0.5 * middle.n,
)

# Add all layers and connections to the network.
network.add_layer(inpt, name="Input Layer")
network.add_layer(middle, name="Hidden Layer")
network.add_layer(out, name="Output Layer")
network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

# Load the Breakout environment.
# NẾU chạy headless/SSH thì đổi render_mode=None hoặc "rgb_array" cho đỡ lỗi SDL
environment = GymEnvironment("BreakoutDeterministic-v4", render_mode="human")
environment.reset()

# Build pipeline from specified components.
environment_pipeline = EnvironmentPipeline(
    network,
    environment,
    encoding=bernoulli,
    action_function=select_softmax,
    output="Output Layer",
    time=100,
    history_length=1,
    delta=1,
    plot_interval=1,
    render_interval=1,
)

def save_checkpoint(network, episode, path=CHECKPOINT_PATH):
    """Lưu trạng thái network + số episode vào file."""
    state = {
        "episode": episode,
        "network_state": network.state_dict(),
    }
    torch.save(state, path)
    print(f"[Checkpoint] Saved at episode {episode} -> {path}")

def load_checkpoint_if_exists(network, path=CHECKPOINT_PATH):
    """Nếu đã có checkpoint thì load vào network và trả về episode bắt đầu."""
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        network.load_state_dict(state["network_state"])
        start_ep = state.get("episode", 0)
        print(f"[Checkpoint] Loaded from {path}, start from episode {start_ep}")
        return start_ep
    else:
        return 0

def run_pipeline(pipeline, episode_count, start_episode=0,
                 checkpoint_path=None, checkpoint_interval=100,
                 training=True):
    """
    start_episode: episode số mấy (dùng khi resume)
    checkpoint_path: nếu khác None thì sẽ lưu
    training=True: dùng để chỉ log cho giai đoạn train/test
    """
    phase = "Train" if training else "Test"
    for i in range(episode_count):
        current_episode = start_episode + i
        total_reward = 0
        pipeline.reset_state_variables()
        is_done = False

        while not is_done:
            result = pipeline.env_step()
            pipeline.step(result)

            reward = result[1]
            total_reward += reward
            is_done = result[2]

        print(f"[{phase}] Episode {current_episode} total reward: {total_reward}")

        # Lưu checkpoint mỗi checkpoint_interval episode (chỉ trong train)
        if training and checkpoint_path is not None:
            if (current_episode + 1) % checkpoint_interval == 0:
                save_checkpoint(pipeline.network, current_episode + 1, checkpoint_path)


# ------------ CHẠY TRAIN + TEST CÓ RESUME ------------

# enable MSTDP
environment_pipeline.network.learning = True

# Nếu đã train trước đó → load checkpoint và tiếp tục train
start_episode = load_checkpoint_if_exists(environment_pipeline.network,
                                          CHECKPOINT_PATH)

print("Training: ")
# ví dụ: tiếp tục train thêm 100 episode nữa
run_pipeline(
    environment_pipeline,
    episode_count=100,
    start_episode=start_episode,
    checkpoint_path=CHECKPOINT_PATH,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    training=True,
)

# stop MSTDP
environment_pipeline.network.learning = False

print("Testing: ")
# Test thì không cần lưu checkpoint, vẫn có thể bắt đầu từ episode 0
run_pipeline(
    environment_pipeline,
    episode_count=100,
    start_episode=0,
    checkpoint_path=None,   # không lưu
    training=False,
)
