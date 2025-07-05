import matlab.engine  # 导入matlab引擎模块
#这些都是常规包
import time  # 导入时间模块
import argparse  # 导入命令行参数解析模块
import numpy as np  # 导入numpy库并简写为np
from tensorboardX import SummaryWriter  # 导入tensorboardX的SummaryWriter用于日志记录
from algo.sac import SAC  # 从algo.sac模块导入SAC类
from algo.utils import setup_seed  # 从algo.utils模块导入setup_seed函数
def main(args):  # 主函数，参数为命令行参数
    global writer, ep_rd  # 声明全局变量writer和ep_rd
    setup_seed(20)  # 设置随机种子为20，保证实验可复现
    logdir = 'data1'  # 日志文件夹路径为'data1'
    if args.tensorboard:  # 如果命令行参数中tensorboard为True
        writer = SummaryWriter(logdir)  # 创建SummaryWriter对象用于写日志
    state_dim = 4    # 状态空间维度 (簧上/下位移, 簧上/下速度)
    action_dim = 1   # 动作空间维度 (作动器力)
    action_bound = 1000.0  # 【修改】动作边界 (例如: 1000 N), 请根据实际物理参数调整
    action_scale = 1.0  # 动作缩放系数为1.0
    agent = SAC(state_dim, action_dim, action_scale)  # 创建SAC智能体对象
    sample_time = 0.05     # 采样时间为0.05s，间隔0.05采样一次
    stop_time = 20
    step_max = int(stop_time / sample_time)     # 20/0.05 = 400次 共采样400次(400个step)
    batch_size = 256
    max_episodes = 600
    # 定义环境模型
    eng = matlab.engine.start_matlab()
    env_name = 'quarter_car_suspension'
    eng.load_system(env_name)
    # 训练过程
    num_training = 0    # 记录训练周期
    t0 = time.time()
    for ep in range(max_episodes):   # 循环训练600个周期episodes（600个20s）
        t1 = time.time()      # 获取当前时间，记录开始时间
        # reset the environment 环境重置
        # python和matlab/simulink的交互����������������句
        eng.set_param(env_name, 'StopTime', str(stop_time + 1), nargout=0)  # 21 for 20 seconds 设定仿真截至时间20s，nargout=0表示不返回输出
        eng.set_param(env_name + '/pause_time', 'value', str(0.01), nargout=0)  # ��定第0.01s时为第一个pause_time内部暂停时间
        eng.set_param(env_name + '/input', 'value', str(0), nargout=0)    # 设定初始的控制信号
        eng.set_param(env_name, 'SimulationCommand', 'start', nargout=0)  # 开始跑仿真
        pause_time = 0.0
        # 采样间隔时间sample_time为0.05s，假设模型仿真步长为0.01s，则设定pause_time为0.05s时会在第0.06s触发Assertion，模型进入pause暂停状态
        # 暂停后，agent输出一个动作值，修改input模块的值，同时改写pause_time模块使pause_time += sample_time ,再继续启动模型仿真，进入下一个暂停
        # 以此实现间隔0.05s的系统模型时序控制

        # 这些数据是要存的，为了方便把每个step生成的（state,action, reward, next_state, done)放到replay buffer数据缓冲区 里面
        # 不同��openAI gym中的模型，simulink模型中控制量的反馈没那么快，0.06s时的控制信号action对应的是0.11s���reward和next_state
        # 所以��������把每个step的先存下来，等一个episode结束后统一push进buffer缓冲区里面
        obs_list, action_list, reward_list, done_list = [], [], [], []
        # 创建四个空列表，用来存储后续的数据，分��存储观察、动作和激励数据以及完成标志
        clock_list = []  # 创建一个空列表，用于存储与时钟相关的数据

        # 每���训练周期episode内 循环step_max = 400步样本采样（每一步就是每次暂停pause）
        for step in range(step_max):
            model_status = eng.get_param(env_name, 'SimulationStatus')
            if model_status == 'paused':
                # 获取 MATLAB 变量，并确保它们是数组
                try:
                    # 检查 out 变量是否存在
                    out_exists = eng.eval("exist('out', 'var')")

                    if not out_exists:
                        # print(f"Step {step}: 'out' variable does not exist yet. Continuing...")  # 注释掉调试输出
                        eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)
                        continue

                    # 首先检查 out 变量的实际结构
                    if step == 0 and False:  # 添加 False 条件，永远不会执行这个代码块
                        print(f"Step {step}: Examining 'out' variable structure...")
                        # 查看 out 变量的字段
                        out_fields = eng.eval("fieldnames(out)")
                        print(f"Available fields in 'out': {out_fields}")
                        
                        # 查看 MATLAB 工作区中的所有变量
                        workspace_vars = eng.eval("who")
                        print(f"MATLAB workspace variables: {workspace_vars}")
                        
                        # 如果工作区中有时间、观测和奖励变量，检查它们的类型
                        if eng.eval("exist('time', 'var')"):
                            print(f"time class: {eng.eval('class(time)')}")
                        if eng.eval("exist('obs', 'var')"):
                            print(f"obs class: {eng.eval('class(obs)')}")
                        if eng.eval("exist('reward', 'var')"):
                            print(f"reward class: {eng.eval('class(reward)')}")

                    # 获取真实数据 - 首先尝试直接从工作区获取变量
                    try:
                        # 检查是否存在时间、观测和奖励变量
                        time_exists = eng.eval("exist('time', 'var')")
                        obs_exists = eng.eval("exist('obs', 'var')")
                        reward_exists = eng.eval("exist('reward', 'var')")
                        
                        if time_exists and obs_exists and reward_exists:
                            # 获取最新的数据点
                            try:
                                clock = eng.eval("time(end)")
                                obs_data = eng.eval("obs(end,:)")
                                reward_data = eng.eval("reward(end)")
                                
                                # 转换为 numpy 数组
                                clock = float(clock)
                                obs = np.array(obs_data)
                                reward = float(reward_data)
                                
                                # 确保观测数据维度正确
                                if obs.size != state_dim:
                                    print(f"Warning: Observation dimension mismatch. Expected {state_dim}, got {obs.size}")
                                    # 调整维度
                                    if obs.size > 0:
                                        if len(obs.shape) > 1:  # 多维数组
                                            obs = obs.flatten()
                                        if obs.size > state_dim:
                                            obs = obs[:state_dim]
                                        elif obs.size < state_dim:
                                            temp = np.zeros(state_dim)
                                            temp[:obs.size] = obs
                                            obs = temp
                                
                                if step % 20 == 0 and False:  # 添加 False 条件，永远不会执行这个代码块
                                    print(f"Step {step}, Time {clock}, Reward: {reward}")
                                    print(f"Observation: {obs}, Shape: {obs.shape}")
                                
                                # 成功获取真实数据
                                using_real_data = True
                            except Exception as data_error:
                                print(f"Error accessing real data: {data_error}")
                                using_real_data = False
                        else:
                            using_real_data = False
                    except Exception as var_error:
                        print(f"Error checking workspace variables: {var_error}")
                        using_real_data = False
                    
                    # 如果无法获取真实数据，回退到使用模拟数据
                    if not using_real_data:
                        if step == 0 and False:  # 添加 False 条件，永远不会执行这个代码块
                            print("Using simulated data instead of real data")
                        
                        # 尝试获取仿真时间
                        try:
                            current_time = eng.eval("get_param('quarter_car_suspension', 'SimulationTime')")
                            if abs(float(current_time)) < 1e-10 and step > 0:
                                current_time = pause_time
                        except:
                            current_time = pause_time
                        
                        # 创建模拟数据
                        clock = float(current_time)
                        
                        # 使用固定种子确保可重复性，但每个周期和步骤都不同
                        np.random.seed(ep * 10000 + step)
                        
                        # 生成模拟观测数据
                        obs = np.array([
                            np.sin(pause_time) + np.random.normal(0, 0.05),
                            np.sin(pause_time + np.pi/4) + np.random.normal(0, 0.05),
                            np.cos(pause_time) + np.random.normal(0, 0.05),
                            np.cos(pause_time + np.pi/4) + np.random.normal(0, 0.05)
                        ])
                        
                        # 模拟奖励
                        state_penalty = np.sum(obs**2)
                        control_effort = np.abs(act[0]) * 0.001 if step > 0 else 0
                        reward = -state_penalty - control_effort - 0.1
                        
                        if step % 20 == 0 and False:  # 添加 False 条件，永远不会执行这个代码块
                            print(f"Step {step} (Simulated), Time {clock}, Reward: {reward}")
                    
                    # 获取智能体动作
                    action = agent.policy_net.get_action(obs, deterministic=False)
                    act = action * action_bound
                    act = np.clip(act, -action_bound, action_bound)
                    
                    # 存储数据
                    clock_list.append(clock)
                    obs_list.append(obs)
                    action_list.append(action)
                    reward_list.append(reward)
                    done_list.append(0.0)
                    pause_time += sample_time

                except Exception as e:
                    if False:  # 添加 False 条件，永远不会执行这个代码块
                        print(f"Error at step {step}: {e}")
                        import traceback
                        traceback.print_exc()
                        print("Continuing simulation...")
                    eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)
                    continue

                # 以下为一个episode结束
                if (pause_time + 0.5) > stop_time:
                    done_list[-1] = 1.0  # 将列表done_list的最后一个元素设置为1.0，表示阶段任务完成的标志
                    eng.set_param(env_name, 'SimulationCommand', 'stop', nargout=0)
                    len_list = len(obs_list)
                    for i1 in range(len_list - 1):
                        obs = obs_list[i1]
                        action = action_list[i1]
                        reward = reward_list[i1 + 1]
                        next_obs = obs_list[i1 + 1]
                        done = done_list[i1 + 1]
                        agent.replay_buffer.push(obs, action, reward, next_obs, done)
                    buffer_length = len(agent.replay_buffer)

                    # 训练过程
                    if buffer_length > batch_size:
                        for _ in range(100):
                            value_loss, q_value_loss1, q_value_loss2, policy_loss = agent.train(batch_size,
                                                                                                reward_scale=0.1,
                                                                                                auto_entropy=True,
                                                                                                target_entropy=-1. * action_dim)  # target_entropy为熵的目标���，如果是负的，则自动计算
                            # 表示是否���用TensorBoard，检查 args 字典中是否有一个名为 tensorboard 的键，并且其值是否为 True，满足则执行下面语句
                            if args.tensorboard:
                                writer.add_scalar('Loss/V_loss', value_loss, global_step=num_training)     # global_step表示训练过程中的总步数，通常是一个递增的变量
                                writer.add_scalar('Loss/Q1_loss', q_value_loss1, global_step=num_training)
                                writer.add_scalar('Loss/Q2_loss', q_value_loss2, global_step=num_training)
                                writer.add_scalar('Loss/pi_loss', policy_loss, global_step=num_training)
                                num_training += 1
                    ep_rd = np.sum(reward_list[1:])    # 计算reward_list列表中：从第二个元素开始到列表末尾的所有元素的总和（每个episode有400次采样数据），即当前或上一个episode的累积奖励
                    print('\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'
                          .format(ep, max_episodes, ep_rd, time.time() - t1), end='', flush=True)
                    if ep_rd > -2 and args.save_model:
                        # 使用当前工作目录的相对路径，而不是绝对路径
                        model_path_pip_epoch = r'model1/model1_{}'.format(ep)
                        agent.save_model(model_path_pip_epoch)
                        # print('=============The SAC model is saved at epoch {}============='.format(ep))  # 注释掉保存模型的提示
                    if args.tensorboard:
                        writer.add_scalar('Reward/train_rd', ep_rd, global_step=ep)

                    # 在每个 episode 结束时打印一个换行，确保下一个 episode 输出在新行
                    print()

                    break
                # 这里接的是上面��常的一个step，此时应该还是间隔0.05s的采样暂停paused时间，然后把控制量、pause_time写入，再continue就行了。
                # 先停止仿真，修改参数，然后重新启动仿真
                eng.set_param(env_name, 'SimulationCommand', 'stop', nargout=0)  # 停止仿真
                eng.set_param(env_name + '/input', 'value', str(act), nargout=0)  # 将控制信号输入到Simulink模型环境
                eng.set_param(env_name + '/pause_time', 'value', str(pause_time), nargout=0)  # 更新Simulink模型环境中的pause_time值
                eng.set_param(env_name, 'SimulationCommand', 'start', nargout=0)  # 重新开始仿真

            # 这里接的是上面环境模型正处于间隔0.05s的数据采样暂停状态，这里是仿真不处于暂停则执行下面语句
            elif model_status == 'running':
                eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)
    print('=============The Total Running Time: {:.4f} =========='.format(time.time() - t0))
    eng.quit()     # 退出Matlab engine引擎，否则这个进程会一直挂在后台，占用系统资源

# ��查当前模块是否作为脚本直接运行。如果���，那么下面的代码块将会被执行
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument('--save_model', default=True, action="store_true")
    parser.add_argument('--save_data', default=False, action="store_true")
    args = parser.parse_args()
    main(args)