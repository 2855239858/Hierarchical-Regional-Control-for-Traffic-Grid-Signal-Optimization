"""Multi-agent traffic light example (single shared policy)."""

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.envs.multiagent import MyMultiTrafficLightGridPOEnv
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env

# Experiment parameters
N_ROLLOUTS = 20  # number of rollouts per training iteration
N_CPUS = 3  # number of parallel workers

# Environment parameters
HORIZON = 200  # time horizon of a single rollout
V_ENTER = 30  # enter speed for departing vehicles
INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
LONG_LENGTH = 100  # length of final edge in route
SHORT_LENGTH = 300  # length of edges that vehicles start on
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 0, 0, 0, 0

EDGE_INFLOW = 300  # inflow rate of vehicles at every edge
N_ROWS = 2 # number of row of bidirectional lanes
N_COLUMNS = 2  # number of columns of bidirectional lanes
Agent_NUM = N_ROWS * N_COLUMNS


# we place a sufficient number of vehicles to ensure they confirm with the
# total number specified above. We also use a "right_of_way" speed mode to
# support traffic light compliance
vehicles = VehicleParams()
num_vehicles = (N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS
vehicles.add(
	veh_id="human",
	acceleration_controller=(SimCarFollowingController, {}),
	car_following_params=SumoCarFollowingParams(
		min_gap=2.5,
		max_speed=V_ENTER,
		decel=7.5,  # avoid collisions at emergency stops
		speed_mode="right_of_way",
	),
	routing_controller=(GridRouter, {}),
	num_vehicles=num_vehicles)

# inflows of vehicles are place on all outer edges (listed here)
outer_edges = []
outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
outer_edges += ["right0_{}".format(i) for i in range(N_COLUMNS)]
outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

# equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
inflow = InFlows()
for edge in outer_edges:
	inflow.add(
		veh_type="human",
		edge=edge,
		vehs_per_hour=500,#600
		#probability=0.25,
		departLane="free",
		departSpeed=V_ENTER)

myNetParams = NetParams(
		inflows=inflow,
		additional_params={
			"speed_limit": V_ENTER + 5,  # inherited from grid0 benchmark
			"grid_array": {
				"short_length": SHORT_LENGTH,
				"inner_length": INNER_LENGTH,
				"long_length": LONG_LENGTH,
				"row_num": N_ROWS,
				"col_num": N_COLUMNS,
				"cars_left": N_LEFT,
				"cars_right": N_RIGHT,
				"cars_top": N_TOP,
				"cars_bot": N_BOTTOM,
			},
			"horizontal_lanes": 1,
			"vertical_lanes": 1,
		},
	)

flow_params = dict(
	# name of the experiment
	exp_tag="grid_0_{}x{}_i{}_multiagent".format(N_ROWS, N_COLUMNS, EDGE_INFLOW),

	# name of the flow environment the experiment is running on
	env_name=MyMultiTrafficLightGridPOEnv,

	# name of the network class the experiment is running on
	network=TrafficLightGridNetwork,

	# simulator that is used by the experiment
	simulator='traci',

	# sumo-related parameters (see flow.core.params.SumoParams)
	sim=SumoParams(
		restart_instance=True,
		sim_step=1,
		render=False,
	),

	# environment related parameters (see flow.core.params.EnvParams)
	env=EnvParams(
		horizon=HORIZON,
		warmup_steps=0,
		sims_per_step=3,
		additional_params={
			"target_velocity": 50,
			"switch_time": 3,
			"num_observed": 10,
			"discrete": False,
			"tl_type": "controlled",
			"num_local_edges": 4,
			"num_local_lights": 4,
		},
	),

	# network-related parameters (see flow.core.params.NetParams and the
	# network's documentation or ADDITIONAL_NET_PARAMS component)
	net=myNetParams,

	# vehicles to be placed in the network at the start of a rollout (see
	# flow.core.params.VehicleParams)
	veh=vehicles,

	# parameters specifying the positioning of vehicles upon initialization
	# or reset (see flow.core.params.InitialConfig)
	initial=InitialConfig(
		spacing='custom',
		shuffle=True,
	),
)

#############################以下为训练部分#################################
def cover_actions(c_a, s_a):
	# 感觉这种cover方式下subagent训练来没什么用啊
	for i in range(Agent_NUM):
		if i != c_a:
			s_a[i] = 0

	# for i in range(Agent_NUM):
	# 	if i == c_a:
	# 		s_a[i] = 1

	
	return s_a


def data_collection(env, vels, queues):
	vehicles = env.k.vehicle
	veh_speeds = vehicles.get_speed(vehicles.get_ids())
	v_temp = 0
	if np.isnan(np.mean(veh_speeds)):
		v_temp = 0
	else:
		v_temp = np.mean(veh_speeds)
	vels.append(v_temp)
	queued_vels = len([v for v in veh_speeds if v < 1])
	queues.append(queued_vels)
	return vels, queues

def normalize_formation(state,Agent_NUM):
	_state = [[] for i in range(Agent_NUM)]
	for i in range(Agent_NUM):
		_state[i] = state["center"+str(i)]
		#_state[i] = state["subagent"+str(i)] # 报错
	return _state


def record_line(log_path, line):
	with open(log_path, 'a') as fp:
		fp.writelines(line)
		fp.writelines("\n")
	return True


if __name__ == "__main__":

		myTrafficNet = TrafficLightGridNetwork(
			name = 'grid',
			vehicles =  vehicles,
			net_params = myNetParams,
		)
		env = MyMultiTrafficLightGridPOEnv(
			env_params=flow_params['env'], sim_params=flow_params['sim'], network=myTrafficNet
		)

		# Perparations for agents
		from flow.core.ppo_agent import *
		# used for ploting
		import matplotlib.pyplot as plt
		import numpy as np

		Reward_num = 1	#0代表多个rewards，1代表1个
		NAME = '2x2_600_PPO_Hierarchy_SOFT_MASTER_try1'

		#*********************
		Epoch = 100	#101
		centre_train_epi = 20	#20
		steps = 210	# 210
		#*********************

		# 画图用的参数
		#横轴
		plot_epoch_checks = []
		#纵轴
		plot_queues_mean = []
		plot_queues_std = []
		plot_speeds_mean = []
		plot_speeds_std = []
		plot_returns_mean = []
		plot_returns_std = []

		sub_agents = [PPO(s_dim=138, a_dim=2, name="subagent" + str(i)) for i in range(Agent_NUM)]
		centre_agent = PPO(s_dim=Agent_NUM, a_dim=Agent_NUM+1, name="centre")
		#centre_agent = PPO(s_dim=Agent_NUM, a_dim=Agent_NUM+1, name="centre_maml")	# 这里命名方式要改变。因为是继承MAML的模型

		# 使用上次训练好的模型
		pretrained_ep_subagent = '90'
		[sub_agents[k].restore_params('sub_agent_{}'.format(str(k)), pretrained_ep_subagent) for k in range(Agent_NUM)]
		#pretrained_ep_master = '0'
		#centre_agent.restore_params('centre_agent_{}'.format(pretrained_ep_master))

		# master可以调用maml参数
		pretrained_ep_master_maml = '20'
		#centre_agent.restore_params('master_maml_multiInflow', pretrained_ep_master_maml, "maml")
		#centre_agent.restore_params('master_maml_multiLen', pretrained_ep_master_maml)
		#centre_agent.restore_params('master_maml_multiLen&Inflow', pretrained_ep_master_maml)

		global_counter = 0	# 记录episode次数
		each_line_path = "collected_data/hierarchy/{}_plot_log.txt".format(NAME)
		test_epoch_path = "collected_data/hierarchy/{}_epoch_log.txt".format(NAME)

		# EPOCH (在这里的作用主要是检测阶段性成果的...
		for ep in range(Epoch):
			
			print("")
			print("sub-agent EPOCH: ", ep)

			################################################################################################
			# episodes for training
			for j in range(centre_train_epi):#centre_train_epi 3
				print('master episode：', j)
				global_counter += 1
				state = env.reset()
				state = normalize_formation(state,Agent_NUM)

				ep_r = 0

				#step
				for step in range(steps):
					actions = [sub_agents[k].choose_action(state[k]) for k in range(Agent_NUM)]
					print("subagent_actions: ", actions)
					actions = np.array(actions)

					# 表示子agent隐藏层的粘合向量，节约计算
					hiddens = [sub_agents[k].get_state(state[k]) for k in range(Agent_NUM)]
					concat_hiddens = np.array(hiddens).reshape(-1, 4)
					# 这种观测了subagent观测到的state
					centre_action = centre_agent.choose_action(concat_hiddens[0])
					print("centre_action: ", centre_action)

					# # 这种只观测了subagent的行为
					# centre_action = centre_agent.choose_action(actions)


					# 输出master监管后的动作
					rl_actions = cover_actions(centre_action, actions)
					print('rl_actions: ', rl_actions)

					next_state, rewards, done, _ = env.step(rl_actions)
					ep_r += rewards

					#centre_agent.experience_store(actions, centre_action, rewards)
					centre_agent.experience_store(concat_hiddens[0], centre_action, rewards)
					
					state = next_state
					state = normalize_formation(state, Agent_NUM)

					if (step + 1) % BATCH == 0 or step == EP_LEN - 1:
						centre_agent.trajction_process(actions)
						centre_agent.update()
						centre_agent.empty_buffer()

					_done = True
					for w in range(Agent_NUM):
						_done *= done["center"+str(w)]

					if _done:
						break
				print('centre steps mean rewards: ', ep_r)
				# centre_agent.summarize(ep_r,  j + (ep * centre_train_epi), 'reward')
				# centre_agent.summarize(ep_r, global_counter, 'Global reward')


			# 每10个epoch保存一次参数
			if ep % 10 == 0:
				#centre_agent.save_params('master', ep)
				pass

		#######################################################################################

			# 每训练完一个epoch就测试一下性能
			if ep >= 0:
				print('test-phase epoch：', ep)
				record_line(each_line_path, "*** Epoch: {} ***\n".format(ep))
				queue, speed, ret = [], [], []
				test_episode = 3
				for i in range(test_episode):
					ep_r, ep_q, ep_v = [], [], []
					state = env.reset()
					state = normalize_formation(state,Agent_NUM)
					
					for step in range(steps):

						data_collection(env, ep_v, ep_q)
						actions = [sub_agents[k].choose_action(state[k]) for k in range(Agent_NUM)]


						# 这种观测了subagent观测到的state
						# 表示子agent隐藏层的粘合向量，节约计算
						hiddens = [sub_agents[k].get_state(state[k]) for k in range(Agent_NUM)]
						concat_hiddens = np.array(hiddens).reshape(-1, 4)	# 是否影响？
						centre_action = centre_agent.choose_action(concat_hiddens[0])
						
						# # 这种只观测了subagent的行为
						# actions = np.array(actions)
						# centre_action = centre_agent.choose_action(actions)

						# 输出master监管后的动作
						rl_actions = cover_actions(centre_action, actions)
						print('rl_actions: ', rl_actions)

						# steps
						next_state, rewards, done, _ = env.step(rl_actions)

						ep_r.append(rewards)

						state = next_state
						state = normalize_formation(state,Agent_NUM)

						_done = True
						for w in range(Agent_NUM):
							_done *= done["center"+str(w)]

						if _done:
							break

					queue.append(np.array(ep_q).mean())
					speed.append(np.array(ep_v).mean())
					ret.append(np.array(ep_r).mean())

					record_line(each_line_path, "Queue: " + str(ep_q) + "\n")
					record_line(each_line_path, "Speed: " + str(ep_v) + "\n")
					record_line(each_line_path, "Return: " + str(ep_r) + "\n")
				

				# record...
				queue_mean = np.array(queue).mean()
				queue_std = np.array(queue).std()
				speed_mean = np.array(speed).mean()
				speed_std = np.array(speed).std()
				return_mean = np.array(ret).mean()
				return_std = np.array(ret).std()

				print("******************** Epoch: {} ********************\n".format(ep))
				print("| Queue: {}, std: {} |".format(queue_mean, queue_std))
				print("| Speed: {}, std: {} |".format(speed_mean, speed_std))
				print("| Return: {}, std: {} |".format(return_mean, return_std))
				print("***************************************************\n")
				record_line(test_epoch_path, "******************** Epoch: {} ********************\n".format(ep))
				record_line(test_epoch_path, "| Queue: {}, std: {} |".format(queue_mean, queue_std))
				record_line(test_epoch_path, "| Speed: {}, std: {} |".format(speed_mean, speed_std))
				record_line(test_epoch_path, "| Return: {}, std: {} |".format(return_mean, return_std))
				record_line(test_epoch_path, "***************************************************\n")

				# plot data collecting
				#横轴
				plot_epoch_checks.append(ep)

				#纵轴
				plot_queues_mean.append(queue_mean)
				plot_queues_std.append(queue_std)
				plot_speeds_mean.append(speed_mean)
				plot_speeds_std.append(speed_std)
				plot_returns_mean.append(return_mean)
				plot_returns_std.append(return_std)

		# plot...
		plt.figure()

		p1 = plt.subplot(321)
		plt.xlabel("Epoch")
		plt.ylabel("plot_queues_mean")
		p1.plot(plot_epoch_checks, plot_queues_mean, linewidth=1)
		p2 = plt.subplot(322)
		plt.xlabel("Epoch")
		plt.ylabel("plot_queues_std")
		p2.plot(plot_epoch_checks, plot_queues_std, linewidth=1)
		p3 = plt.subplot(323)
		plt.xlabel("Epoch")
		plt.ylabel("plot_speeds_mean")
		p3.plot(plot_epoch_checks, plot_speeds_mean, linewidth=1)
		p4 = plt.subplot(324)
		plt.xlabel("Epoch")
		plt.ylabel("plot_speeds_std")
		p4.plot(plot_epoch_checks, plot_speeds_std, linewidth=1)
		p5 = plt.subplot(325)
		plt.xlabel("Epoch")
		plt.ylabel("plot_returns_mean")
		p5.plot(plot_epoch_checks, plot_returns_mean, linewidth=1)
		p6 = plt.subplot(326)
		plt.xlabel("Epoch")
		plt.ylabel("plot_returns_std")
		p6.plot(plot_epoch_checks, plot_returns_std, linewidth=1)

		#plt.savefig('/test_{}.png'.format(NAME))
		plt.show()