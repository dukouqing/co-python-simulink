import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from algo.utils import ReplayBuffer, SoftQNetwork, PolicyNetwork


class SAC():
    def __init__(self, state_dim, action_dim, action_range):
        self.num_training = 0  # 记录训练的步数
        hidden_dim = 128   # 神经网络隐藏层的维度

        self.replay_buffer_size = 10000  # 经验回放缓冲区的大小
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)  # 创建一个经验回放缓冲区实例
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)  # 创建两个Q网络实例，用于近似状态-动作值函数
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)   # 创建两个目标软Q网络实例，用于跟踪Q网络的目标值
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)  # 创建一个策略网络实例，用于生成动作
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)  # 创建一个对数变量，用于平衡探索和利用
        # print('Soft Q Network (1,2): ', self.soft_q_net1)
        # print('Policy Network: ', self.policy_net)
        # 将Q网络的参数复制到目标Q网络的参数中，初始化目标网络
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)
        # 创建均方误差损失函数实例
        self.soft_q_criterion1 = nn.MSELoss()  # 定义两个MSELoss（均方误差损失函数）的实例，MSE是衡量预测值和真实值之间差异的一种方式(预测值 - 真实值)²
        self.soft_q_criterion2 = nn.MSELoss()  # 用于比较软目标网络（Soft Q-function）的输出与硬目标网络（Hard Q-function）的输出之间的差异
        # 在SAC算法中，软目标网络通常用于估计Q值，并且有两个版本（或“副本”）的Q函数，即Q1和Q2。
        # 使用两个不同的损失函数（soft_q_criterion1和soft_q_criterion2）来计算两个Q函数之间的差异，有助于提高学习过程的稳定性。
        # 每个损失函数可以分别用于计算Q1与Q2之间的差异，或者用于计算Q函数的预测值与实际观察到的回报之间的差异。

        # 定义学习率
        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4
        # 创建优化器来优化网络参数
        # 设置了四个优化器，每个优化器分别用于更新三个不同的网络（两个Soft Q网络和一个策略网络）以及一个用于控制熵项的参数alpha。
        # 每个优化器都使用不同的学习率来调整各自网络的参数，以便在训练过程中优化性能。
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr) # 创建了一个优化器实例，用于更新soft_q_net1网络的参数。
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        # self.log_alpha是存储alpha对数值的变量。self.log_alpha通常被优化以最大化alpha，从而最大化熵项
        # [self.log_alpha]表明优化器将仅优化self.log_alpha这一个参数
    def train(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        # 定义train方法，该方法接受多个参数，包括批量大小batch_size、奖励缩放因子reward_scale、自动熵参数auto_entropy、目标熵target_entropy、折扣因子gamma和软更新系数soft_tau
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)  # 从经验回放缓冲区中随机抽取一批数据
        # print('sample:', state, action,  reward, done)
        # 将抽取的数据转换为PyTorch张量，并确保它们在正确的设备上（GPU或CPU）运行
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)  # unsqueeze(1) 用于将奖励从标量转换为维度为 [batch_size, 1] 的张量
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)   # 使用两个软Q网络分别预测当前状态和动作的Q值。
        predicted_q_value2 = self.soft_q_net2(state, action)

        # 使用策略网络评估当前状态和下一个状态，以生成新的动作 new_action 和对应的对数概率 log_prob
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)

        # 标准化奖励值，使用批量均值和标准差进行归一化，并添加一个很小的数值以防止数值问题
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)

        # alpha = 0.0 探索(最大熵)和开发(最大Q)之间的权衡
        # 如果自动熵auto_entropy为真，则更新温度参数alpha，它用于平衡探索和利用
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()  # 计算alpha_loss
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()   # 清除梯度，将梯度置零，是反向传播前的一个必要步骤，用于准备优化器以进行新的梯度更新
            alpha_loss.backward()              # 执行反向传播，计算alpha_loss相对于模型参数的梯度
            self.alpha_optimizer.step()        # 使用计算出的梯度来更新模型参数，即更新alpha
            self.alpha = self.log_alpha.exp()  # 将log_alpha（alpha的对数）转换为alpha的值，因为self.alpha通常是alpha的实际值，而不是对数形式
        else:     # 如果auto_entropy不是True，这段代码会执行
            self.alpha = 1.0      # 将alpha设置为1.0，这意味着不使用自动熵
            alpha_loss = 0        # 将alpha_loss设置为0，表示在这种情况下不需要计算alpha损失

        # Training Q Function
        # 计算了两个软目标网络（self.target_soft_q_net1 和 self.target_soft_q_net2）在下一个状态和相应的新动作下的输出，然后取这两个输出的最小值
        # 这个最小值减去 self.alpha * next_log_prob 是为了将熵项纳入目标Q值中
        # self.alpha 是一个温度参数，通常用于控制熵项在总损失中的作用
        # next_log_prob 是动作的概率的对数，通常用于熵损失计算
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),
                                 self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        # 计算目标Q值（target_q_value），它包含了奖励和折扣的未来回报
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        # 计算预测Q值和目标Q值之间的损失，分别使用两个不同的损失函数实例
        # .detach() 用于阻止梯度反向传播到target_q_value，因为它是计算得到的值，不希望它的梯度影响网络的更新
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
        # 更新软目标网络的参数
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # 计算策略网络的损失
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
        # 更新策略网络的参数
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # self.writer.add_scalar('Loss/Alpha_loss', alpha_loss, global_step=self.num_training)
        # self.writer.add_scalar('Loss/Q1_loss', q_value_loss1, global_step=self.num_training)
        # self.writer.add_scalar('Loss/Q2_loss', q_value_loss2, global_step=self.num_training)
        # self.writer.add_scalar('Loss/pi_loss', policy_loss, global_step=self.num_training)
        # self.num_training +=1

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net更新目标网络的参数
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return alpha_loss, q_value_loss1, q_value_loss2, policy_loss  # 返回了多个损失值

    def save_model(self, path):   # 定义了一个名为 save_model 的方法，它接收一个参数 path，这个参数表示保存模型的路径
        # 使用 torch.save 函数保存 soft_q_net1（一个Q网络）的状态字典（state_dict）到指定路径下的 q1.pth 文件中
        # state_dict 包含了模型中所有参数及其当前值的字典
        torch.save(self.soft_q_net1.state_dict(), path + 'q1.pth')
        torch.save(self.soft_q_net2.state_dict(), path + 'q2.pth')
        torch.save(self.policy_net.state_dict(), path + 'policy.pth')
        # torch.save(self.target_soft_q_net1.state_dict(), path + 'q1_target.pth')
        # torch.save(self.target_soft_q_net2.state_dict(), path + 'q2_target.pth')
        print('=============The SAC model is saved=============')


    def load_model(self, path):  # 定义了一个名为 load_model 的方法，它接收一个参数 path
        # 使用 load_state_dict 方法加载 soft_q_net1 的参数，这些参数是从 path + 'q1.pth' 文件中读取的
        self.soft_q_net1.load_state_dict(torch.load(path + 'q1.pth'))
        self.soft_q_net2.load_state_dict(torch.load(path + 'q2.pth'))
        self.policy_net.load_state_dict(torch.load(path + 'policy.pth'))
        self.target_soft_q_net1 = deepcopy(self.soft_q_net1)
        self.target_soft_q_net2 = deepcopy(self.soft_q_net2)
        # 使用 deepcopy 函数复制 soft_q_net1 和 soft_q_net2 到 target_soft_q_net1 和 target_soft_q_net2。
        # 在SAC中，这些目标网络是用于更新策略网络的Q值估计的，因此需要确保它们是Q网络参数的精确副本。


        # self.target_soft_q_net1.load_state_dict(torch.load(path + 'q1_target.pth'))
        # self.target_soft_q_net2.load_state_dict(torch.load(path + 'q2_target.pth'))

        # self.soft_q_net1.eval()
        # self.soft_q_net2.eval()
        # self.policy_net.eval()
















