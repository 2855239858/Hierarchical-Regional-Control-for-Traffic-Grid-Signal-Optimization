import tensorflow as tf
import numpy as np

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.002
C_LR = 0.002
BATCH = 30
A_UPDATE_STEPS = 5
C_UPDATE_STEPS = 10 #10
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

NUM_TASKS = 21
A_grads_collec = [[] for i in range(A_UPDATE_STEPS)]    # A_UPDATE_STEPS * NUM_TASKS    (5,1)

# 原始的PPO_agent
class PPO(object):

    def __init__(self, s_dim=32, a_dim=1, name="meme"):
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.name = name
            self.buffer_a = []
            self.buffer_s = []
            self.buffer_r = []
            self.global_steps = 0
            self.update_steps_a = 0
            self.update_steps_c = 0
            self.global_counter = 0

            # critic
            with tf.variable_scope(self.name + '_critic'):
                l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
                self.v = tf.layers.dense(l1, 1) # 添加一个全连接层（？）
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))

                # 设置Adam更新critic
                c_optimizer = tf.train.AdamOptimizer(C_LR)
                self.ctrain_op = c_optimizer.minimize(self.closs)

            # actor
            self.pi, pi_params = self._build_anet(self.name + '_pi', trainable=True)
            self.oldpi, oldpi_params = self._build_anet(self.name + '_oldpi', trainable=False)

            self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
            pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)  # shape=(None, )
            oldpi_prob = tf.gather_nd(params=self.oldpi, indices=a_indices)  # shape=(None, )
            ratio = pi_prob / (oldpi_prob + 1e-8)
            surr = ratio * self.tfadv  # surrogate loss

            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                surr,
                tf.clip_by_value(ratio, 1. - 0.2, 1. + 0.2) * self.tfadv))

            # 用来记录优化总步数
            self.global_steps = tf.Variable(0, trainable=False) 
            # 设置Adam更新actor
            a_optimizer = tf.train.AdamOptimizer(A_LR)
            self.atrain_op = a_optimizer.minimize(self.aloss, global_step=self.global_steps)


            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter("baseline/ppo/" + self.name + "_log/", self.sess.graph)
            #self.saver = tf.train.Saver(max_to_keep=20)

            # restore MAML
            vl = [v for v in tf.global_variables() if "Adam" not in v.name]
            self.saver = tf.train.Saver(var_list=vl)


    def update(self):
        s = np.vstack(self.buffer_s)
        r = np.array(self.buffer_r)[:, np.newaxis]
        a = self.buffer_a
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        actor_loss = self.sess.run(self.aloss, {self.tfs: s, self.tfa: a, self.tfadv: adv})
        self.summarize(actor_loss,self.global_counter,'Actor_loss')

        # 在每一个s-a-r-step进行参数更新
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]
        # print("actor global steps: ", self.sess.run(self.global_steps))

        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss,self.global_counter,'Critic_loss')

        # update critic
        # 在每一个s-a-r-step进行参数更新
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        # print("critic global steps: ", self.sess.run(self.global_steps))
        self.global_counter += 1

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            self.l1 = tf.layers.dense(self.tfs, 32, tf.nn.relu,  trainable=trainable)
            self.l2 = tf.layers.dense(self.l1,  32, tf.nn.relu,  trainable=trainable)
            self.out = tf.layers.dense(self.l2, self.a_dim, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return self.out, params

    def display_prob(self,s):
        prob = self.sess.run(self.out, feed_dict={self.tfs: s[None, :]})
        print(prob)


    def choose_action(self, s):

        # subagent观测的s是一个size为138的向量
        # center观测的s是一个size为4的向量
        prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        
        return action

    def get_state(self, s):
        s = s[np.newaxis, :]
        h = self.sess.run(self.l2, {self.tfs: s})[0]
        return h

    def get_v(self, s):
        _s = np.array(s)
        if _s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def experience_store(self, s, a, r):
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r)

    def empty_buffer(self):
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []

    # 每一步的reward进行一个discount，让越远的reward影响变小
    def trajction_process(self, s_):
        v_s_ = self.get_v(s_)
        discounted_r = []
        for r in self.buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)

        discounted_r.reverse()
        self.buffer_r = discounted_r

    def summarize(self, reward, i, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=reward)
        self.writer.add_summary(summary, i)
        self.writer.flush()

    def save_params(self,name,ep):
        save_path = self.saver.save(self.sess,'my_net/ppo/{}_ep{}.ckpt'.format(name,ep))
        print("Save to path:",save_path)
    def restore_params(self,name,ep,ppo_type="normal"):
        if(ppo_type == "noramal"):
            self.saver.restore(self.sess,'my_net/ppo/{}_ep{}.ckpt'.format(name,ep))
            print("Restore params from: ", 'my_net/ppo/{}_ep{}.ckpt'.format(name,ep))
        
        # 调用MAML参数
        if(ppo_type == "maml"):
            self.saver.restore(self.sess,'my_net/maml/{}_ep{}.ckpt'.format(name,ep))
            print("Restore params from: ", 'my_net/maml/{}_ep{}.ckpt'.format(name,ep))

# 用来训练为MAML搜集梯度数据的PPO_agent
class PPO_FOR_MAML(object):

    def __init__(self, s_dim=32, a_dim=1, name="meme"):
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.name = name
            self.buffer_a = []
            self.buffer_s = []
            self.buffer_r = []
            self.global_steps = 0
            self.update_steps_a = 0
            self.update_steps_c = 0
            self.global_counter = 0

            # critic
            with tf.variable_scope(self.name + '_critic'): 
                l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
                self.v = tf.layers.dense(l1, 1)
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                # self.advantage = self.v - self.tfdc_r
                self.closs = tf.reduce_mean(tf.square(self.advantage))
                # 设置Adam更新critic
                self.c_optimizer = tf.train.AdamOptimizer(C_LR)
                # maml 设置梯度计算。为了获取梯度需要拆成两步
                #self.c_gradient_compu = self.c_optimizer.compute_gradients(self.closs, pi_params)    # 不加参数会报错
                #self.c_gradient_compu = self.c_optimizer.compute_gradients(self.closs)    # 但是critic无pi_params
                # 把梯度从返回结果中整理出来
                #self.c_grads_holder = [(tf.placeholder(tf.float32), v) for (g, v) in self.c_gradient_compu]
                #self.c_optm = self.c_optimizer.apply_gradients(self.c_grads_holder) # 进行梯度更新
                self.ctrain_op = self.c_optimizer.minimize(self.closs)

            # actor
            self.pi, pi_params = self._build_anet(self.name + '_pi', trainable=True)
            self.oldpi, oldpi_params = self._build_anet(self.name + '_oldpi', trainable=False)

            self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
            pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)  # shape=(None, )
            oldpi_prob = tf.gather_nd(params=self.oldpi, indices=a_indices)  # shape=(None, )
            ratio = pi_prob / (oldpi_prob + 1e-8)
            surr = ratio * self.tfadv  # surrogate loss

            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                surr,
                tf.clip_by_value(ratio, 1. - 0.2, 1. + 0.2) * self.tfadv))

            # 用来记录优化总步数
            self.global_steps = tf.Variable(0, trainable=False) 
            # 设置Adam更新actor
            self.a_optimizer = tf.train.AdamOptimizer(A_LR)
            # maml 设置梯度计算。为了获取梯度需要拆成两步
            self.a_gradient_compu = self.a_optimizer.compute_gradients(self.aloss, pi_params)
            # 把梯度从返回结果(g, v)中整理出来。但是会报错（格式问题
            #self.a_gradient_compu = [(tf.placeholder(tf.float32, shape=g.get_shape())) for (g, v) in self.a_gradient_compu]
            #self.a_gradient_compu = [(tf.placeholder(tf.float32, shape=g.get_shape()), v) for (g, v) in self.a_gradient_compu]
            self.a_optm = self.a_optimizer.apply_gradients(self.a_gradient_compu) # 进行梯度更新
            # self.atrain_op = a_optimizer.minimize(self.aloss, global_step=self.global_steps)


            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter("baseline/ppo/" + self.name + "_log/", self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=20)


    def update(self):
        s = np.vstack(self.buffer_s)
        r = np.array(self.buffer_r)[:, np.newaxis]
        a = self.buffer_a
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

        # update actor
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        actor_loss = self.sess.run(self.aloss, {self.tfs: s, self.tfa: a, self.tfadv: adv})
        self.summarize(actor_loss,self.global_counter,'Actor_loss')

        # 在每一个MDP进行参数更新
        # 如果训练maml，这里需要搜集actor梯度信息
        for i in range(A_UPDATE_STEPS):
            # 搜集该步update的梯度信息...写入缓存...
            A_grad = self.sess.run(self.a_gradient_compu, {self.tfs: s, self.tfa: a, self.tfadv: adv})
            A_grads_collec[i].append(A_grad)
            #print("appended shape: ",  np.array(A_grads_collec).shape)
            self.sess.run(self.a_optm, {self.tfs: s, self.tfa: a, self.tfadv: adv})
            #self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv})




        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss,self.global_counter,'Critic_loss')

        # update critic
        # 在每一个MDP进行参数更新
        # 如果训练maml，这里需要搜集critic梯度信息
        for i in range(C_UPDATE_STEPS):
            # 搜集该步update的梯度信息...写入缓存...
            #C_grad = self.sess.run(self.c_gradient_compu, {self.tfs: s, self.tfdc_r: r})
            #C_grads_collec[i].append(C_grad)
            #self.sess.run(self.c_optm, {self.tfs: s, self.tfdc_r: r})
            self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})

        #print("global counter: ", self.sess.run(self.global_counter))
        
        self.global_counter += 1


    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            self.l1 = tf.layers.dense(self.tfs, 32, tf.nn.relu,  trainable=trainable)
            self.l2 = tf.layers.dense(self.l1,  32, tf.nn.relu,  trainable=trainable)
            self.out = tf.layers.dense(self.l2, self.a_dim, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return self.out, params

    def display_prob(self,s):
        prob = self.sess.run(self.out, feed_dict={self.tfs: s[None, :]})
        print(prob)


    def choose_action(self, s):

        # subagent观测的s是一个size为138的向量
        # center观测的s是一个size为4的向量
        prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob

        # print("from choose_action_function: ", action, self.name)
        # s = np.array(s)
        # print(s.shape)
        
        return action

    def get_state(self, s):
        s = s[np.newaxis, :]
        h = self.sess.run(self.l2, {self.tfs: s})[0]
        return h

    def get_v(self, s):
        _s = np.array(s)
        if _s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def experience_store(self, s, a, r):
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r)

    def empty_buffer(self):
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []

    # 每一步的reward进行一个discount，让越远的reward影响变小
    def trajction_process(self, s_):
        v_s_ = self.get_v(s_)
        discounted_r = []
        for r in self.buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)

        discounted_r.reverse()
        self.buffer_r = discounted_r

    def summarize(self, reward, i, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=reward)
        self.writer.add_summary(summary, i)
        self.writer.flush()

    # def save_params(self,name,ep):
    #     save_path = self.saver.save(self.sess,'my_net/ppo/{}_ep{}.ckpt'.format(name,ep))
    #     print("Save to path:",save_path)
    # def restore_params(self,name,ep):
    #     self.saver.restore(self.sess,'my_net/ppo/{}_ep{}.ckpt'.format(name,ep))
    #     print("Restore params from: ", 'my_net/ppo/{}_ep{}.ckpt'.format(name,ep))



# 使用maml来训练的元PPO_agent
# 它和原来的master同步训练，需要与环境做互动（获得数据来计算loss之类的）
# 训练过程中，maml和原master共享所有的s,a,r
# 不过更新方式不同……！！
class PPO_MAML(object):

    def __init__(self, s_dim=32, a_dim=1, name="meme"):
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.name = name
            self.buffer_a = []
            self.buffer_s = []
            self.buffer_r = []
            self.global_steps = 0
            self.update_steps_a = 0
            self.update_steps_c = 0
            self.global_counter = 0
            # MAML


            # critic
            with tf.variable_scope(self.name + '_critic'):
                l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
                self.v = tf.layers.dense(l1, 1)
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                # self.advantage = self.v - self.tfdc_r
                self.closs = tf.reduce_mean(tf.square(self.advantage))
                # 设置Adam更新critic
                self.c_optimizer = tf.train.AdamOptimizer(C_LR)
                self.ctrain_op = self.c_optimizer.minimize(self.closs)

            # actor
            self.pi, pi_params = self._build_anet(self.name + '_pi', trainable=True)
            self.pi_params = pi_params  # 复制出来备用于其他函数
            self.oldpi, oldpi_params = self._build_anet(self.name + '_oldpi', trainable=False)

            self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
            pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)  # shape=(None, )
            oldpi_prob = tf.gather_nd(params=self.oldpi, indices=a_indices)  # shape=(None, )
            ratio = pi_prob / (oldpi_prob + 1e-8)
            surr = ratio * self.tfadv  # surrogate loss

            # 这个对maml作用不大
            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                surr,
                tf.clip_by_value(ratio, 1. - 0.2, 1. + 0.2) * self.tfadv))

            # 用来记录优化总步数
            self.global_steps = tf.Variable(0, trainable=False)

            # 设置Adam更新actor
            self.a_optimizer = tf.train.AdamOptimizer(A_LR)
            # 只是为了取一个形状
            self.grads_shape = self.a_optimizer.compute_gradients(self.aloss, pi_params)
            self.grads_shape = np.array(self.grads_shape)
            print("grads_shape ready...")
            print(self.grads_shape) # 这是梯度列表list，里面是g, v组合
            
            #################################################
            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter("baseline/maml/" + self.name + "_log/", self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=20)



    def init_maml(self):

        # 把梯度格式从返回结果中整理出来
        # 必须要在更新后才能整理，不然list里无内容

        # 6*2(placeholer) 注意feed的是placeholder，它是(1,1)--(...)
        self.a_gradient_compu = [(tf.placeholder(tf.float32, shape=g.get_shape()), v) for (g, v) in self.grads_shape]
        #self.a_gradient_compu = tf.placeholder(tf.float32, shape=self.grads_shape.shape) # 报错
        print("TENSORS: ", self.a_gradient_compu)
        self.a_optm = self.a_optimizer.apply_gradients(self.a_gradient_compu)

        self.sess.run(tf.global_variables_initializer())

    def update(self):

        self.global_counter += 1

        s = np.vstack(self.buffer_s)
        r = np.array(self.buffer_r)[:, np.newaxis]
        a = self.buffer_a
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        # 作为maml, placeholder里的观测值似乎没什么用啊。因为梯度已经计算出来了
        actor_loss = self.sess.run(self.aloss, {self.tfs: s, self.tfa: a, self.tfadv: adv}) # 这个其实不用算..用来看效果
        self.summarize(actor_loss,self.global_counter,'Actor_loss')

        # 各任务梯度求和
        A_grads_collec_np = np.array(A_grads_collec)
        A_grads_collec_np_sum = A_grads_collec_np.sum(axis=1) # 包括5步梯度
        #print("grad_sum_shape: ", A_grads_collec_np_sum[0].shape) #(6,2)
        #print("grad_sum[0]: ", A_grads_collec_np_sum[0])
        # 在每一个MDP进行参数更新
        for i in range(A_UPDATE_STEPS):
            # 从缓存中读出梯度，计算梯度和并运用...
            self.sess.run(self.a_optm, 
            feed_dict={self.a_gradient_compu[0][0] : A_grads_collec_np_sum[i][0][0],
            self.a_gradient_compu[1][0] : A_grads_collec_np_sum[i][1][0],
            self.a_gradient_compu[2][0] : A_grads_collec_np_sum[i][2][0],
            self.a_gradient_compu[3][0] : A_grads_collec_np_sum[i][3][0],
            self.a_gradient_compu[4][0] : A_grads_collec_np_sum[i][4][0],
            self.a_gradient_compu[5][0] : A_grads_collec_np_sum[i][5][0]
            })

        print("MAML updated for one step.")
        #A_grads_collec = [[] for i in range(A_UPDATE_STEPS)]    # 清空梯度数组

        print("actor global steps: ", self.sess.run(self.global_steps))


        # update critic
        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss,self.global_counter,'Critic_loss')

        # 在每一个MDP进行参数更新
        for i in range(C_UPDATE_STEPS):
            self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})

        print("critic global steps: ", self.sess.run(self.global_steps))

        

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            self.l1 = tf.layers.dense(self.tfs, 32, tf.nn.relu,  trainable=trainable)
            self.l2 = tf.layers.dense(self.l1,  32, tf.nn.relu,  trainable=trainable)
            self.out = tf.layers.dense(self.l2, self.a_dim, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return self.out, params

    def display_prob(self,s):
        prob = self.sess.run(self.out, feed_dict={self.tfs: s[None, :]})
        print(prob)


    # 删除了一些功能

    def experience_store(self, s, a, r):
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r)

    def empty_buffer(self):
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []

    # 删除了一些功能

    def summarize(self, reward, i, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=reward)
        self.writer.add_summary(summary, i)
        self.writer.flush()

    def save_params(self,name,ep):
        save_path = self.saver.save(self.sess,'my_net/maml/{}_ep{}.ckpt'.format(name,ep))
        print("Save to path:",save_path)

    # def restore_params(self,name,ep):
    #     self.saver.restore(self.sess,'my_net/maml/{}_ep{}.ckpt'.format(name,ep))
    #     print("Restore params from: ", 'my_net/maml/{}_ep{}.ckpt'.format(name,ep))


class PPO_MAML_AC(object):

    def __init__(self, s_dim=32, a_dim=1, name="meme"):
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.name = name
            self.buffer_a = []
            self.buffer_s = []
            self.buffer_r = []
            self.global_steps = 0
            self.update_steps_a = 0
            self.update_steps_c = 0
            self.global_counter = 0
            # MAML


            # critic
            with tf.variable_scope(self.name + '_critic'):
                l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
                self.v = tf.layers.dense(l1, 1)
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                # self.advantage = self.v - self.tfdc_r
                self.closs = tf.reduce_mean(tf.square(self.advantage))
                # 设置Adam更新critic
                self.c_optimizer = tf.train.AdamOptimizer(C_LR)
                self.ctrain_op = self.c_optimizer.minimize(self.closs)

            # actor
            self.pi, pi_params = self._build_anet(self.name + '_pi', trainable=True)
            self.pi_params = pi_params  # 复制出来备用于其他函数
            self.oldpi, oldpi_params = self._build_anet(self.name + '_oldpi', trainable=False)

            self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
            pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)  # shape=(None, )
            oldpi_prob = tf.gather_nd(params=self.oldpi, indices=a_indices)  # shape=(None, )
            ratio = pi_prob / (oldpi_prob + 1e-8)
            surr = ratio * self.tfadv  # surrogate loss

            # 这个对maml作用不大
            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                surr,
                tf.clip_by_value(ratio, 1. - 0.2, 1. + 0.2) * self.tfadv))

            # 用来记录优化总步数
            self.global_steps = tf.Variable(0, trainable=False)

            # 设置Adam更新actor
            self.a_optimizer = tf.train.AdamOptimizer(A_LR)
            # 只是为了取一个形状
            self.grads_shape = self.a_optimizer.compute_gradients(self.aloss, pi_params)
            self.grads_shape = np.array(self.grads_shape)
            print("grads_shape ready...")
            print(self.grads_shape) # 这是梯度列表list，里面是g, v组合
            
            #################################################
            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter("baseline/maml/" + self.name + "_log/", self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=20)



    def init_maml(self):

        # 把梯度从返回结果中整理出来
        # 必须要在5步更新后才能整理，不然list里无内容

        # 6*2(placeholer) 注意feed的是placeholder，它是(1,1)--(...)
        self.a_gradient_compu = [(tf.placeholder(tf.float32, shape=g.get_shape()), v) for (g, v) in self.grads_shape]
        #self.a_gradient_compu = tf.placeholder(tf.float32, shape=self.grads_shape.shape) # 报错
        print("TENSORS: ", self.a_gradient_compu)
        self.a_optm = self.a_optimizer.apply_gradients(self.a_gradient_compu)

        self.sess.run(tf.global_variables_initializer())

    def update(self):

        self.global_counter += 1

        s = np.vstack(self.buffer_s)
        r = np.array(self.buffer_r)[:, np.newaxis]
        a = self.buffer_a
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful



        # update actor
        # 作为maml, placeholder里的观测值似乎没什么用啊。因为梯度已经计算出来了
        actor_loss = self.sess.run(self.aloss, {self.tfs: s, self.tfa: a, self.tfadv: adv}) # 这个其实不用算..用来看效果
        self.summarize(actor_loss,self.global_counter,'Actor_loss')

        # 各任务梯度求和
        A_grads_collec_np = np.array(A_grads_collec)
        A_grads_collec_np_sum = A_grads_collec_np.sum(axis=1) # 包括5步梯度
        print("grad_sum_shape: ", A_grads_collec_np_sum[0].shape) #(6,2)
        # 在每一个MDP进行参数更新
        for i in range(A_UPDATE_STEPS):
            # 从缓存中读出梯度，计算梯度和并运用...
            self.sess.run(self.a_optm, 
            feed_dict={self.a_gradient_compu[0][0] : A_grads_collec_np_sum[i][0][0],
            self.a_gradient_compu[1][0] : A_grads_collec_np_sum[i][1][0],
            self.a_gradient_compu[2][0] : A_grads_collec_np_sum[i][2][0],
            self.a_gradient_compu[3][0] : A_grads_collec_np_sum[i][3][0],
            self.a_gradient_compu[4][0] : A_grads_collec_np_sum[i][4][0],
            self.a_gradient_compu[5][0] : A_grads_collec_np_sum[i][5][0]
            })

        print("actor global steps: ", self.sess.run(self.global_steps))




        # update critic
        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss,self.global_counter,'Critic_loss')

        # 在每一个MDP进行参数更新
        for i in range(C_UPDATE_STEPS):
            # 从缓存中读出梯度，计算梯度和并运用...(critic暂时不用)
            self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})

        print("critic global steps: ", self.sess.run(self.global_steps))

        

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            self.l1 = tf.layers.dense(self.tfs, 32, tf.nn.relu,  trainable=trainable)
            self.l2 = tf.layers.dense(self.l1,  32, tf.nn.relu,  trainable=trainable)
            self.out = tf.layers.dense(self.l2, self.a_dim, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return self.out, params

    def display_prob(self,s):
        prob = self.sess.run(self.out, feed_dict={self.tfs: s[None, :]})
        print(prob)


    # 删除了一些功能

    def experience_store(self, s, a, r):
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r)

    def empty_buffer(self):
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []

    # 删除了一些功能

    def summarize(self, reward, i, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=reward)
        self.writer.add_summary(summary, i)
        self.writer.flush()

    def save_params(self,name,ep):
        save_path = self.saver.save(self.sess,'my_net/maml/{}_ep{}.ckpt'.format(name,ep))
        print("Save to path:",save_path)
    # def restore_params(self,name,ep):
    #     self.saver.restore(self.sess,'my_net/maml/{}_ep{}.ckpt'.format(name,ep))
    #     print("Restore params from: ", 'my_net/maml/{}_ep{}.ckpt'.format(name,ep))



