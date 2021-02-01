"""Multi-agent traffic light example (single shared policy)."""
"""这个代码主要用来训练subagent"""
"""master的训练以及最后的测试结果主要见Hierarchy_master.py"""

# 第一次subagent训练完毕 -2020.6.12 epoch90

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
		vehs_per_hour=600,
		# probability=0.25,
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
	for i in range(Agent_NUM):
		if i != c_a:
			s_a[i] = 0
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
		#_state[i] = state["subagent"+str(i)]
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

		Reward_num = 0	#0代表多个rewards，1代表1个
		NAME = '2x2_600_PPO_Hierarchy_SOFT_try0'

		#*********************
		Epoch = 100	#100 10
		sub_train_epi = 5	# 20 1
		steps = 210	# 210 10

		delay = 0
		delay_step = 3
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

		pretrained_ep = '0'
		#[sub_agents[k].restore_params('sub_agent_{}'.format(str(k)), pretrained_ep) for k in range(Agent_NUM)]
		
		# train
		global_counter = 0	# 记录episode次数
		each_line_path = "collected_data/hierarchy/{}_plot_log.txt".format(NAME)
		test_epoch_path = "collected_data/hierarchy/{}_epoch_log.txt".format(NAME)

		# EPOCH (在这里的作用主要是检测阶段性成果的...
		for ep in range(Epoch):
			print("")
			print("sub-agent EPOCH: ", ep)
			
			# training episodes
			for i in range(sub_train_epi):
				
				print("sub-agent episode: ", i)
				global_counter += 1

				# 获得state[k]数组,获得的名称竟然是center...不过没有影响
				state = env.reset()
				state = normalize_formation(state, Agent_NUM)

				# 记录每一个episode的累加reward

				# 多reward
				ep_r = [0] * Agent_NUM

				# # 单reward
				# ep_r = 0

				# steps
				for step in range(steps):
					# print("sub-agent step: ",step)
					actions = [sub_agents[k].choose_action(state[k]) for k in range(Agent_NUM)]
					
					next_state, rewards, done, _ = env.step(actions)	# dict, float, dict
					#next_state, rewards, done, _ = env.step(233)	# for traceback

					# 对该episode的reward进行累加

					# 多reward
					for k in range(Agent_NUM):
						ep_r[k] += rewards[k]
					#print("*****ep_r: ", ep_r)

					# 单reward
					#ep_r += rewards

					for k in range(Agent_NUM):
						# 把数据存入该对象的buffer，用作训练数据
						sub_agents[k].experience_store(state[k], actions[k], rewards[k])
						#sub_agents[k].experience_store(state[k], actions[k], rewards)

					state = next_state

					state = normalize_formation(state,Agent_NUM)
					if (step + 1) % BATCH == 0 or  step == EP_LEN - 1:
						print("!!!BATCH!!!")
						# count_store = 0
						for k in range(Agent_NUM):
							sub_agents[k].trajction_process(state[k])
							sub_agents[k].update()	# 每30steps进行一次更新
							sub_agents[k].empty_buffer()	# 每30steps清空缓存(s, r, a)


					_done = True
					for k in range(Agent_NUM):
						_done *= done["center"+str(k)]

					if _done:
						break

				#print('{} subagent steps total rewards:'.format(NAME), ep_r)
				#print('{} subagent steps total rewards:'.format(NAME), sum(ep_r))

				# [sub_agents[k].summarize(ep_r[k], i + (ep * sub_train_epi), 'reward') for k in range(Agent_NUM)]
				# sub_agents[0].summarize(sum(ep_r),i + (ep * sub_train_epi), 'total reward')
				# centre_agent.summarize(sum(ep_r)/Agent_NUM, i + (ep * sub_train_epi), 'mean reward')
				# [sub_agents[k].summarize(ep_r, i + (ep * sub_train_epi), 'total reward') for k in range(Agent_NUM)]
				# centre_agent.summarize(ep_r/Agent_NUM, i + (ep * sub_train_epi), 'mean reward')


			if ep % 10 == 0:
				[sub_agents[k].save_params('sub_agent_{}'.format(str(k)),ep) for k in range(Agent_NUM)]
				#pass


#######################################################################################


			# testing episodes(no master)
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

						# steps
						next_state, rewards, done, _ = env.step(actions)

						# 求该step四个agent的总reward
						reward_sum = 0
						for i in rewards:
							reward_sum = reward_sum + rewards[i]	# 字典取值
						ep_r.append(reward_sum)
						
						# 单reward
						#ep_r.append(rewards)
						
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

		#plt.savefig('/test_{}.png'.format(NAME))	# 无权限
		plt.show()
