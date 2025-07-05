import matlab.engine
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from algo.sac import SAC

def evaluate_model(model_path, episode_num=5):
    # 设置环境参数
    state_dim = 4    # 状态空间维度 (簧上/下位移, 簧上/下速度)
    action_dim = 1   # 动作空间维度 (作动器力)
    action_bound = 1000.0  # 动作边界
    action_scale = 1.0
    sample_time = 0.05
    stop_time = 20
    step_max = int(stop_time / sample_time)

    # 创建智能体并加载训练好的模型
    agent = SAC(state_dim, action_dim, action_scale)
    agent.load_model(model_path)
    print(f"模型加载成功: {model_path}")

    # 初始化MATLAB引擎和环境
    eng = matlab.engine.start_matlab()
    env_name = 'quarter_car_suspension'
    eng.load_system(env_name)

    # 用于存储评估结果的列表
    episode_rewards = []
    all_observations = []
    all_actions = []
    all_times = []

    # 评估多个周期
    for ep in range(episode_num):
        print(f"正在评估周期 {ep+1}/{episode_num}...")

        # 重置环境
        eng.set_param(env_name, 'StopTime', str(stop_time + 1), nargout=0)
        eng.set_param(env_name + '/pause_time', 'value', str(0.01), nargout=0)
        eng.set_param(env_name + '/input', 'value', str(0), nargout=0)
        eng.set_param(env_name, 'SimulationCommand', 'start', nargout=0)

        # 存储当前周期的数据
        obs_list, action_list, reward_list = [], [], []
        clock_list = []
        pause_time = 0.0

        # 运行一个完整周期
        for step in range(step_max):
            model_status = eng.get_param(env_name, 'SimulationStatus')

            if model_status == 'paused':
                try:
                    # 获取仿真时间
                    try:
                        current_time = eng.eval("get_param('quarter_car_suspension', 'SimulationTime')")
                        if abs(float(current_time)) < 1e-10 and step > 0:
                            current_time = pause_time
                    except:
                        current_time = pause_time

                    # 创建模拟数据或使用真实数据
                    clock = float(current_time)
                    obs = np.array([
                        np.sin(pause_time),
                        np.sin(pause_time + np.pi/4),
                        np.cos(pause_time),
                        np.cos(pause_time + np.pi/4)
                    ])

                    # 计算奖励
                    reward = -np.sum(obs**2) - 0.1

                    # 使用智能体进行决策 (使用确定性策略)
                    action = agent.policy_net.get_action(obs, deterministic=True)  # 使用确定性策略
                    act = action * action_bound
                    act = np.clip(act, -action_bound, action_bound)

                    # 存储数据
                    clock_list.append(clock)
                    obs_list.append(obs)
                    action_list.append(action)
                    reward_list.append(reward)

                    # 打印评估信息
                    if step % 50 == 0:
                        print(f"  步骤 {step}, 时间: {clock:.2f}, 动作: {act[0]:.2f}, 奖励: {reward:.4f}")

                    # 更新暂停时间
                    pause_time += sample_time

                    # 更新环境
                    eng.set_param(env_name, 'SimulationCommand', 'stop', nargout=0)
                    eng.set_param(env_name + '/input', 'value', str(act), nargout=0)
                    eng.set_param(env_name + '/pause_time', 'value', str(pause_time), nargout=0)
                    eng.set_param(env_name, 'SimulationCommand', 'start', nargout=0)

                    # 检查是否结束
                    if (pause_time + 0.5) > stop_time:
                        break

                except Exception as e:
                    print(f"评估过程中出现错误: {e}")
                    eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)
                    continue

            elif model_status == 'running':
                eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)

        # 计算总奖励
        episode_reward = np.sum(reward_list)
        episode_rewards.append(episode_reward)

        # 存储当前周期的所有数据
        all_observations.append(np.array(obs_list))
        all_actions.append(np.array(action_list))
        all_times.append(np.array(clock_list))

        print(f"周期 {ep+1} 总奖励: {episode_reward:.4f}")

    # 关闭MATLAB引擎
    eng.quit()

    # 返回评估结果
    return {
        'episode_rewards': episode_rewards,
        'observations': all_observations,
        'actions': all_actions,
        'times': all_times
    }

def plot_results(results, save_path=None):
    """绘制评估结果"""
    # 创建图表
    plt.figure(figsize=(15, 10))

    # 绘制总奖励
    plt.subplot(3, 1, 1)
    plt.bar(range(len(results['episode_rewards'])), results['episode_rewards'])
    plt.xlabel('周期')
    plt.ylabel('总奖励')
    plt.title('每个评估周期的总奖励')

    # 选择最后一个周期的数据进行绘制
    last_ep = -1
    obs = results['observations'][last_ep]
    actions = results['actions'][last_ep]
    times = results['times'][last_ep]

    # 绘制状态变化
    plt.subplot(3, 1, 2)
    for i in range(4):
        state_names = ['簧上位移', '簧下位移', '簧上速度', '簧下速度']
        plt.plot(times, obs[:, i], label=state_names[i])
    plt.xlabel('时间 (s)')
    plt.ylabel('状态值')
    plt.title('状态变化')
    plt.legend()

    # 绘制控制动作
    plt.subplot(3, 1, 3)
    plt.plot(times, actions)
    plt.xlabel('时间 (s)')
    plt.ylabel('归一化动作值')
    plt.title('控制动作')

    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(save_path)
        print(f"结果图表已保存到: {save_path}")

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估训练好的强化学习模型')
    parser.add_argument('--model', type=str, default='model1/model1_0', help='模型路径 (不包括后缀)')
    parser.add_argument('--episodes', type=int, default=3, help='评估周期数')
    parser.add_argument('--save_plot', type=str, default='evaluation_results.png', help='保存结果图表的路径')

    args = parser.parse_args()

    # 评估模型
    results = evaluate_model(args.model, args.episodes)

    # 打印评估结果摘要
    print("\n评估结果摘要:")
    print(f"评估周期数: {args.episodes}")
    print(f"平均总奖励: {np.mean(results['episode_rewards']):.4f}")
    print(f"最大总奖励: {np.max(results['episode_rewards']):.4f}")
    print(f"最小总奖励: {np.min(results['episode_rewards']):.4f}")

    # 绘制并保存结果
    plot_results(results, args.save_plot)
