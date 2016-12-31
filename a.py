""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """

from argparse import ArgumentParser
import numpy as np
import cPickle as pickle
import gym


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--nb_hidden_neurons', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=10)
    ap.add_argument('--learning_rate', type=int, default=1e-4)
    ap.add_argument('--gama', type=float, default=0.99)  # discount factor for reward
    ap.add_argument('--decay_rate', type=float, default=0.99)  # decay factor for RMSProp leaky sum of grad^2
    ap.add_argument('--resume', type=int, default=0)  # resume from previous checkpoint?
    ap.add_argument('--render', type=int, default=True)
    ap.add_argument('--input_dim', type=int, default=80 * 80)  # input dimensionality: 80x80 grid
    return ap.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(I):
    """ preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]     # crop
    I = I[::2,::2,0]  # downsample by factor of 2
    I[I == 144] = 0   # erase background (background type 1)
    I[I == 109] = 0   # erase background (background type 2)
    I[I != 0] = 1     # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(model, x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h # return probability of taking action 2, and hidden state


def policy_backward(model,eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}


args = parse_args()
if args.resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(args.nb_hidden_neurons, args.input_dim) / np.sqrt(args.input_dim) # "Xavier" initialization
    model['W2'] = np.random.randn(args.nb_hidden_neurons) / np.sqrt(args.nb_hidden_neurons)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory
env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if args.render:
        env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = preprocess(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(args.input_dim)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(model, x)
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    y = 1 if action == 2 else 0 # a "fake label"
    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

    if done: # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr, args.gamma)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(model, eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % args.batch_size == 0:
            for k,v in model.iteritems():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = args.decay_rate * rmsprop_cache[k] + (1 - args.decay_rate) * g**2
                model[k] += args.learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
