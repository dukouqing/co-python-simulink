import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch库，用于深度学习
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式API
import torch.optim as optim  # 导入优化器
from copy import deepcopy  # 导入深拷贝函数

use_cuda = torch.cuda.is_available()  # 检查是否有可用的GPU
device = torch.device("cuda" if use_cuda else "cpu")  # 设置设备（GPU或CPU）

from algo.utils import ReplayBuffer  # 从utils模块导入经验回放缓冲区

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_range):  # 初始化函数，接收状态维度、动作维度、隐藏层维度和动作范围
        super(Actor, self).__init__()  # 调用父类初始化方法
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 第一个全连接层，从状态到隐藏层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 第二个全连接层，隐藏层到隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 第三个全连接层，隐藏层到动作
        self.action_range = action_range  # 存储动作范围，用于将输出缩放到合适的范围

    def forward(self, state):  # 前向传播函数，接收状态作为输入
        x = F.relu(self.fc1(state))  # 通过第一层并应用ReLU激活函数
        x = F.relu(self.fc2(x))  # 通过第二层并应用ReLU激活函数
        x = torch.tanh(self.fc3(x))  # 通过第三层并应用tanh激活函数，输出范围为[-1,1]
        # 将输出缩放到动作范围
        action = x * self.action_range  # 将[-1,1]范围的输出缩放到实际动作范围
        return action  # 返回动作

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):  # 初始化函数，接收状态维度、动作维度和隐藏层维度
        super(Critic, self).__init__()  # 调用父类初始化方法
        # Q1 网络
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)  # Q1的第一个全连接层，输入为状态和动作的拼接
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Q1的第二个全连接层
        self.fc3 = nn.Linear(hidden_dim, 1)  # Q1的输出层，输出Q值

        # Q2 网络
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)  # Q2的第一个全连接层，结构与Q1相同
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)  # Q2的第二个全连接层
        self.fc6 = nn.Linear(hidden_dim, 1)  # Q2的输出层，输出Q值

    def forward(self, state, action):  # 前向传播函数，接收状态和动作作为输入
        # 拼接状态和动作
        sa = torch.cat([state, action], 1)  # 将状态和动作在第1维度上拼接

        # Q1 值
        q1 = F.relu(self.fc1(sa))  # 通过Q1的第一层并应用ReLU激活函数
        q1 = F.relu(self.fc2(q1))  # 通过Q1的第二层并应用ReLU激活函数
        q1 = self.fc3(q1)  # 通过Q1的输出层

        # Q2 值
        q2 = F.relu(self.fc4(sa))  # 通过Q2的第一层并应用ReLU激活函数
        q2 = F.relu(self.fc5(q2))  # 通过Q2的第二层并应用ReLU激活函数
        q2 = self.fc6(q2)  # 通过Q2的输出层

        return q1, q2  # 返回两个Q网络的输出

    def Q1(self, state, action):  # 只计算Q1值的函数，用于策略更新
        # 只返回Q1值，用于策略更新
        sa = torch.cat([state, action], 1)  # 将状态和动作在第1维度上拼接
        q1 = F.relu(self.fc1(sa))  # 通过Q1的第一层并应用ReLU激活函数
        q1 = F.relu(self.fc2(q1))  # 通过Q1的第二层并应用ReLU激活函数
        q1 = self.fc3(q1)  # 通过Q1的输出层

        return q1  # 返回Q1值

class TD3():
    def __init__(self, state_dim, action_dim, action_range):  # 初始化函数，接收状态维度、动作维度和动作范围
        self.num_training = 0  # 记录训练的步数
        hidden_dim = 128       # 神经网络隐藏层的维度

        self.replay_buffer_size = 10000  # 经验回放缓冲区的大小
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)  # 创建一个经验回放缓冲区实例

        # 创建Actor和Critic网络
        self.actor = Actor(state_dim, action_dim, hidden_dim, action_range).to(device)  # 创建Actor网络并移动到指定设备
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, action_range).to(device)  # 创建Actor目标网络

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)  # 创建Critic网络并移动到指定设备
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)  # 创建Critic目标网络

        # 复制参数到目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())  # ���Actor网络的参数复制到目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())  # 将Critic网络的参数复制到目标网络

        # 定义优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)  # 为Actor网络创建Adam优化器，学习率为3e-4
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)  # 为Critic网���创建Adam优化器，学习率为3e-4

        # TD3特有的超参数
        self.policy_noise = 0.2 * action_range  # 目标策略平滑噪声，动作范围的20%
        self.noise_clip = 0.5 * action_range    # 噪声裁剪范围，动作范围的50%
        self.policy_freq = 2                    # 策略延迟更新频率，每2次Critic更新更新一次Actor

        # 保存动作范围
        self.action_range = action_range  # 存储动作范围，用于后续操作

    def select_action(self, state, deterministic=True):  # 选择动作的函数，接收状态和确定性标志
        """选择动作，用于与环境交互"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 将状态转换为张量并添加批次维度
        with torch.no_grad():  # 不计算梯度
            action = self.actor(state).cpu().numpy().flatten()  # 通过Actor网络获取动作，并转换为NumPy数组

        # 如果不是确定性策略，添加探索噪声
        if not deterministic:  # 如果不是确定性模式（通常用于训练）
            noise = np.random.normal(0, self.action_range * 0.1, size=action.shape)  # 生成随机噪声
            action = np.clip(action + noise, -self.action_range, self.action_range)  # 将噪声添加到动作并裁剪到合法范围

        return action  # 返回选择的动作

    def train(self, batch_size=256, gamma=0.99, tau=0.005):  # 训练函数，接收批量大小、折扣因子和软更新系数
        """训练TD3算法"""
        self.num_training += 1  # 训练步数加1

        # 从经验回放缓冲区中随机抽取一批数据
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)  # 采样一批经验

        # 将抽取的数据转换为PyTorch张量
        state = torch.FloatTensor(state).to(device)  # 将状态转换为张量并移动到设备
        action = torch.FloatTensor(action).to(device)  # 将动作转换为张量并移动到设备
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)  # 将奖励转换为张量，添加维度，并移动到设备
        next_state = torch.FloatTensor(next_state).to(device)  # 将下一状态转换为张量并移动到设备
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)  # 将完成标志转换为张量，添加维度，并移动到设备

        # TD3算法的核心：目标策略平滑
        with torch.no_grad():  # 不计算梯度
            # 选择下一个动作并添加裁剪噪声
            noise = torch.randn_like(action) * self.policy_noise  # 生成与动作形状相同的随机噪声
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)  # 裁剪噪声

            next_action = self.actor_target(next_state) + noise  # 通过目标Actor网络获取下一动作并添加噪声
            next_action = torch.clamp(next_action, -self.action_range, self.action_range)  # 裁剪动作到合法范围

            # 计算目标Q值，取两个Q网络中的最小值
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)  # 通过目标Critic网络获取目标Q值
            target_Q = torch.min(target_Q1, target_Q2)  # 取两个Q值中的最小值，避免过估计
            target_Q = reward + (1 - done) * gamma * target_Q  # 计算目标Q值，使用贝尔曼方程

        # 计算当前Q值
        current_Q1, current_Q2 = self.critic(state, action)  # 通过Critic网络获取当前Q值

        # 计算Critic损失
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)  # 计算两个Q网络的均方误差损失

        # 更新Critic
        self.critic_optimizer.zero_grad()  # 清空Critic优化器的梯度
        critic_loss.backward()  # 反向传播
        self.critic_optimizer.step()  # 更新Critic网络参数

        # 延迟更新策略和目标网络
        if self.num_training % self.policy_freq == 0:  # 如果当前步数是策略更新频率的倍数
            # 计算Actor损失 - 最大化Q值
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()  # 计算Actor损失，最大化Q值

            # 更新Actor
            self.actor_optimizer.zero_grad()  # 清空Actor优化器的梯度
            actor_loss.backward()  # 反向传播
            self.actor_optimizer.step()  # 更新Actor网络参数

            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):  # 遍历Critic网络和目标网络的参数
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)  # 软更新目标Critic网络

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):  # 遍历Actor网络和目标网络的参数
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)  # 软更新目标Actor网络

            return critic_loss.item(), actor_loss.item()  # 返回Critic损失和Actor损失

        return critic_loss.item(), 0  # 如果没有更新Actor，返回Critic损失和0

    def save_model(self, path):  # 保存模型函数，接收保存路径
        """保存模型"""
        torch.save(self.actor.state_dict(), path + 'actor.pth')  # 保存Actor网络参数
        torch.save(self.critic.state_dict(), path + 'critic.pth')  # 保存Critic网络参数
        print('=============The TD3 model is saved=============')  # 打印保存成功信息

    def load_model(self, path):  # 加载模型函数，接收加载路径
        """加载模型"""
        self.actor.load_state_dict(torch.load(path + 'actor.pth'))  # 加载Actor网络参数
        self.critic.load_state_dict(torch.load(path + 'critic.pth'))  # 加载Critic网络参数
        self.actor_target.load_state_dict(self.actor.state_dict())  # 将加载的Actor参数复制到目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())  # 将加载的Critic参数复制到目标网络
