import os
import torch
import numpy as np
import cv2
import sys
import time as t_time

# Import các hàm xử lý ảnh của BindsNet
from bindsnet.datasets.preprocess import binary_image, crop, gray_scale, subsample
from bindsnet.encoding import bernoulli
from bindsnet.environment import GymEnvironment
from bindsnet.learning import MSTDP
from bindsnet.network import Network
from bindsnet.network.nodes import Input, IzhikevichNodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_softmax

# ==========================================
# 1. CẤU HÌNH SIÊU NHẸ (LITE MODE)
# ==========================================
CHECKPOINT_PATH = "breakout_lite.pth" # Đổi tên file save mới
SIMULATION_TIME = 20      # Giảm thời gian suy nghĩ xuống 20ms
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- THIẾT BỊ: {DEVICE} ---")
print("--- CHẾ ĐỘ: LITE (40x40 pixels) ---")

# ==========================================
# 2. TẠO MÔI TRƯỜNG TÙY BIẾN (Custom Environment)
# ==========================================
# Chúng ta ghi đè lớp GymEnvironment để giảm ảnh xuống 40x40
class FastBreakout(GymEnvironment):
    def preprocess(self) -> None:
        # Cắt bỏ phần điểm số, chỉ lấy vùng chơi
        self.obs = crop(self.obs, 34, 194, 0, 160)
        self.obs = gray_scale(self.obs)
        # QUAN TRỌNG: Giảm xuống 40x40 (Thay vì 80x80) -> Nhẹ hơn 4 lần
        self.obs = subsample(self.obs, 40, 40) 
        self.obs = binary_image(self.obs)
        self.obs = torch.from_numpy(self.obs).float()

# ==========================================
# 3. XÂY DỰNG MẠNG (Đã chỉnh theo 40x40)
# ==========================================
network = Network(dt=1.0)

# Input: 40x40 = 1600 nơ-ron (Nhẹ hơn nhiều so với 6400)
# Tắt traces=False ở Input để giảm tải tính toán
inpt = Input(n=40 * 40, shape=[1, 1, 1, 40, 40], traces=False)

# Hidden & Output giữ nguyên
middle = IzhikevichNodes(n=100, traces=True)
out = IzhikevichNodes(n=4, refrac=0, traces=True)

# Kết nối
inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=0.2)
middle_out = Connection(
    source=middle, target=out, wmin=0, wmax=1.0,
    update_rule=MSTDP, nu=1e-2, norm=0.5 * middle.n
)

network.add_layer(inpt, name="Input Layer")
network.add_layer(middle, name="Hidden Layer")
network.add_layer(out, name="Output Layer")
network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")
network.to(DEVICE)

# ==========================================
# 4. KHỞI TẠO PIPELINE
# ==========================================
# Dùng class FastBreakout vừa tạo
environment = FastBreakout("BreakoutDeterministic-v4", render_mode="rgb_array")
environment.reset()

pipeline = EnvironmentPipeline(
    network,
    environment,
    encoding=bernoulli,
    action_function=select_softmax,
    output="Output Layer",
    time=SIMULATION_TIME, 
    history_length=1,
    delta=1,
    plot_interval=None,
    render_interval=None,
)

# ==========================================
# 5. VÒNG LẶP CHÍNH (Đã tối ưu hiển thị)
# ==========================================
pipeline.network.learning = True

# Tạo cửa sổ nhỏ gọn hơn
window_name = "BindsNet Lite (40x40)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 600, 300) 

print("\n--- BẮT ĐẦU ---")
print("Hãy theo dõi Terminal bên dưới xem có dòng 'Step...' nhảy số không.")

episode = 0
try:
    while True:
        total_reward = 0
        pipeline.reset_state_variables()
        done = False
        step_counter = 0
        
        while not done:
            start_time = t_time.time()
            
            # 1. Chạy game
            result = pipeline.env_step()
            
            # 2. In log để biết code không bị treo trước khi tính toán nặng
            sys.stdout.write(f"\rEpisode {episode} | Step {step_counter} | Computing...")
            sys.stdout.flush()

            # 3. Tính toán SNN (Phần nặng nhất)
            pipeline.step(result)
            
            reward = result[1]
            done = result[2]
            info = result[3]
            step_counter += 1

            # 4. HIỂN THỊ
            if "gym_obs" in info:
                # Lấy ảnh 40x40
                raw_img = info["gym_obs"].detach().cpu().squeeze().numpy()
                
                # Input SNN (Delta)
                snn_img = info["delta_obs"].detach().cpu().squeeze().numpy()

                # Phóng to giá trị (0,1 -> 0,255)
                raw_display = (raw_img * 255).astype(np.uint8)
                snn_display = (snn_img * 255).astype(np.uint8)
                
                # Ghép ảnh
                combined = np.hstack((raw_display, snn_display))
                
                # Phóng to kích thước ảnh để dễ nhìn (40x40 bé quá)
                combined_big = cv2.resize(combined, (600, 300), interpolation=cv2.INTER_NEAREST)
                
                # Vẽ thông tin
                cv2.putText(combined_big, f"Step: {step_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150), 2)
                if reward != 0:
                     cv2.putText(combined_big, f"REWARD!", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255), 3)

                cv2.imshow(window_name, combined_big)

            # 5. QUAN TRỌNG: Giúp cửa sổ 'thở'
            # Nếu tính toán mất >0.1s thì cửa sổ sẽ lag, lệnh này giúp Force Update
            cv2.waitKey(1)
            
            total_reward += reward

        print(f"\nEpisode {episode} Done | Total Reward: {total_reward}")
        
        # Lưu mỗi 10 ván
        if (episode + 1) % 10 == 0:
            torch.save(network.state_dict(), CHECKPOINT_PATH)
            print("-> Saved checkpoint.")
            
        episode += 1

except KeyboardInterrupt:
    print("\n--- STOP ---")
    cv2.destroyAllWindows()