import traci
import torch
import numpy as np
import save_and_load as sl

policy = sl.load_model("2023.05.06-17.52.23")  # 输入保存模型的文件夹名称
print("导入模型。")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 以下为测试阶段获取一个state后，产生对应action的代码
state = np.array([1, 1, 1, 1])
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
with torch.no_grad():
    # action是int
    action = policy(state).max(1)[1].view(1, 1).item()

print(action)
