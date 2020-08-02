
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from collections import deque
from tqdm import tqdm
import time
import numpy as np
import random
import timeit
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import joblib
from PER import *
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#%load_ext tensorboard
#import dask.dataframe as dd
#import tracemalloc

#tracemalloc.start() #FFOR MEMORYLEAK TESTING

tf.config.optimizer.set_jit(True)
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# In[3]:

#SETTING GPU AND ENSURE IT USES MOMORY GROWTH
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tf.config.experimental.list_physical_devices()


filename = "data\\GBPJPY_2019_2020_Smart_Diff_RL_V2.npy"
xgb_filename = "Boost\\ID1806_1015_GJ_20pip_sample18_RL_V1.joblib_cv.dat"

SYMBOL = "GBPJPY" #The forex pair used
SYMBOL_FACTOR = 100 #Multiplication factor used for currency pair
SPREAD = 2.5 #Commision costs used
BOOK_NAME = "2015_orderbook"
LOAD_SCALER = False
EPOCHS = 500
REPLAY_MEMORY_SIZE = 1000_000
MIN_REPLAY_MEMORY_SIZE = 100_000
MINIBATCH_SIZE = 64
ENVIROMENT_OBSERVATION_SPACE = 75  #Number of features in dataset + 1 feature of win/loss profit
SCALE_SPACE = 38 #To minimize computation time, one-hot features (0,1) are not scaled.
ACTION_SPACE = (
    4  # The four actions available for the agent (Do nothing, buy, sell, close open order)
)
STACK_SIZE = 6 #Number of samples staqcked together for the lstm

DISCOUNT = 0.90  # gamme
EPISODES = 428 #Episodes in dataset. Here one episode is one trading day of 24 (hours) * 12 * 5 min samples
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.05
REPLAY = []

LOAD_MODEL = False
MODEL_NAME = "7927_XGB76_tau0_001_S6_DDDQN_GJ_LSTM64_256_192_96_D240_Mini64_20K_LR0_1"
MODEL_ID = random.randint(1000,10_000) #Use this to add a 4-digit unique ID to each model run
MODEL_FILENAME = "EPOCH33_tau0_001_S6_DDDQN_XGB77_GJ_LSTM64_256_192_96_D240_Mini64_20K_LR0_1__GBPJPY_7927_EPS3829.h5"
MODEL_PATH = f"models\\{MODEL_FILENAME}"

TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

epsilon = 1
ep_rewards = []
book_time = int(time.time())
learning_rate = 0.1
epoch_count = 0

# TENSORBOARD STUFF
log_dir=f"logs\\{MODEL_NAME}_{MODEL_ID}_{TIME}"
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=2)
file_writer = tf.summary.create_file_writer(log_dir + "\\metrics")

# CREATE FOLDERS IF THEY DOESN'T EXIST
if not os.path.isdir("models"):
    os.makedirs("models")
if not os.path.isdir("scaler"):
    os.makedirs("scaler")
if not os.path.isdir("output"):
    os.makedirs("output")


# SETTING SEED FOR MORE REPETITIVE RESULTS
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


def checkpoint():
    tf.keras.callbacks.ModelCheckpoint(filepath=f"models\\CHK_{MODEL_NAME}__{SYMBOL}_{MODEL_ID}_{epoch_count}.h5",
                                                 save_weights_only=True,
                                                 verbose=1)
 
#ENVIROMENT CLASS. THIS CLASS HOLD ALL INFORMATION ON THE ENVIROMENT, WHICH IN THIS CASE IS ABOUT THE FOREX. EVERYTHING ELSE THAN THE MODEL...
#-ORDER TRACING, REWARDS, GETTING NEW STATES, ETC.
class forex_env():

    def __init__(self):
        self.total_win_loss = 0
        self.win_loss = 0
        self.orders = ""
        self.order_price = 0
        self.prev_win_loss = 0
        self.difference = 0
        self.prev_day = 0
        self.state_deque = deque(maxlen=STACK_SIZE)

    #INITIAL STATE IS FETCHED AND CUSTOM SCALER IS FITTED
    def preprocess(self, filename):

        point_resp = np.reshape([0.0], (-1, 1))
        data = np.load(filename, allow_pickle=True)
    
        win_loss_proxy = np.random.uniform(low=-1.0, high=1.0, size=data.shape[0]).reshape(-1, 1)
        self.xdata = np.concatenate((data, win_loss_proxy), axis=1)
    
        scaler = myscaler.custom_fit(data[:,3:])
    
        with open(f"scaler/{SYMBOL}_scaler_{MODEL_ID}_{TIME}.pkl", "wb") as f:
            pickle.dump(scaler, f)

        for idx in range(STACK_SIZE): #Add samples to stack
            sample = self.xdata[idx,:]
            self.state_deque.append(sample)
    
        layered_state = np.stack(self.state_deque, axis=0)
        self.prev_day = layered_state[STACK_SIZE-1,4]
        print(f'XDATA SHAPE: {self.xdata.shape[0]}')
        return layered_state
    
    #STACK STATES FOR LSTM
    def stack_states(self, new_state):
        
        self.state_deque.append(new_state)
        layered_state = np.stack(self.state_deque, axis=0) 
     
        return layered_state
        
        
    #DATE AND TIME FETCHED FROM CURRENT STATE. IT IS ONLY USED FOR MONITORING PURPOSES (PRINT AND LOG)
    def to_datetime(self, current_state):
        datetime_object = datetime(
            year=2020,
            month=int(current_state[STACK_SIZE-1,3]),
            day=int(current_state[STACK_SIZE-1,4]),
            hour=int(current_state[STACK_SIZE-1,5]),
            minute=int(current_state[STACK_SIZE-1,6]),
        )
        return datetime_object

    #THE ACTION SELECTED IS VALIDATED. EG. YOU CANNOT OPEN A NEW ORDER IF ONE IS ALREADY OPEN.
    def action_check(self, current_state, action, close_market):
        # Check action if its legal and then send it to market or return a legal action if chosen action is illegal.
        # 0 = do nothing   #1 = buy   #2 = sell   #3 = close order
        if close_market:
            if self.orders == "long":
                action = 3
            elif self.orders == "short":
                action = 3
            else:
                action = 0
        else:
            if action == 0:
                pass

            elif action == 1:
                if self.orders == "long":
                    action = 0
                elif self.orders == "short":
                    action = 3

            elif action == 2:
                if self.orders == "long":
                    action = 3
                elif self.orders == "short":
                    action = 0

            elif action == 3:
                if self.orders == "long":
                    action = 3
                elif self.orders == "short":
                    action = 3
                else:
                    action = 0

        return action


    #ONLY USED FOR MONITORING PURPOSES TO TRACK PROGRESS. IT PRINTS AND WRITES LOG
    def book_keeping(self, orders, orderprice, close_price, current_state):

        start_book = timeit.default_timer()
        timestamp = self.to_datetime(current_state)
        time_string = timestamp.strftime("%Y.%m.%d, %H:%M:%S")
        self.total_win_loss += self.win_loss
        
        # Create a numpy array with all the trade/state info for the current state
        print(f'{time_string}, {close_price:6.2f}, {orders:5}, {self.win_loss:7.2f}, {self.total_win_loss:7.2f}')
        trade_info = (np.array([time_string, self.orders, orderprice, close_price, self.win_loss, self.total_win_loss]))
        
        with open(f"output/{BOOK_NAME}_{MODEL_ID}_{book_time}.csv", "a") as f:
            np.savetxt(f, trade_info, fmt="%s", newline=" ", delimiter=",")
            f.write("\n")

        return self.total_win_loss

    #USED TO RESET ENVIRMONET
    def reset(self):
        self.prev_win_loss = 0
        self.win_loss = 0
        self.order_price = None
        self.orders = ''

    #ORDERS/ACTIONS SENT TO MARKET ARE HANDLED HERE
    def to_market(self, current_state, action, timestep):

        current = current_state
        if action == 0:
            if self.orders == 'long':
                self.win_loss = (current[STACK_SIZE-1,0] - self.order_price)
            elif self.orders == 'short':
                self.win_loss = (self.order_price - current[STACK_SIZE-1,0])
            else:
                self.win_loss = 0
            close_order = False
        
        elif action == 1:
            #Send buy order to market
            self.orders = 'long'
            self.order_price =  current[STACK_SIZE-1,0] #Get 'ask' price from current state and saves it

            timestamp = self.to_datetime(current_state)
            time_string = timestamp.strftime("%Y.%m.%d, %H:%M:%S")

            close_order = False
            self.win_loss = 0
        
        elif action == 2:
            #Send sell order to market
            self.orders = 'short'
            self.order_price = current[STACK_SIZE-1,0] #Get 'bid' price from current state and saves it

            timestamp = self.to_datetime(current_state)
            time_string = timestamp.strftime("%Y.%m.%d, %H:%M:%S")

            close_order = False
            self.win_loss = 0
        

        elif action == 3 and self.orders =='long':
            #Send close order to market
            close_price = current[STACK_SIZE-1,0] - (SPREAD/SYMBOL_FACTOR)
            self.win_loss = (close_price - self.order_price)*SYMBOL_FACTOR #close long order
            self.total_win_loss = self.book_keeping(self.orders, self.order_price, close_price, current_state) #Send trade info to CSV file
            close_order = True
            self.orders =''
            self.order_price = 0
            
        
        elif action == 3 and self.orders =='short':
            #Send close order to market
            close_price = current[STACK_SIZE-1,0] + (SPREAD/SYMBOL_FACTOR)
            self.win_loss = (self.order_price - close_price)*SYMBOL_FACTOR #close short order
            self.total_win_loss = self.book_keeping(self.orders, self.order_price, close_price, current_state) #Send trade info to CSV file
            close_order = True
            self.orders =''
            self.order_price = 0
            
 
        #UPDATE TIMESTEP AND FETCH NEW STATE
        timestep += 1
        if timestep == self.xdata.shape[0]-1:
            terminal_state = True
            print(F'TERMINAL STATE! {timestep}')
        else:
            terminal_state = False

        new_state = self.xdata[timestep]
        
        if self.win_loss > 100:
            win_loss_adj = 100
        elif self.win_loss < -100:
            win_loss_adj = -100
        else:
            win_loss_adj = self.win_loss

        #ADD WIN/LOSS AS A FEATURE SO THE MODEL GETS SOME FEEDBACK ON PROGRESS
        wl = np.array(win_loss_adj/100).astype('float64')
        new_state[ENVIROMENT_OBSERVATION_SPACE-1] = wl

        return self.stack_states(new_state), terminal_state, close_order, timestep

    #REWARD CALCULATED
    def get_reward(self, close):
        
        if self.orders == "":
            reward = -0.03
        else:
            self.difference = self.win_loss - self.prev_win_loss
            abs_win_loss = abs(self.win_loss)
            close_bonus = 0
        
            #BONUS FOR CLOSED ORDERS
            if close:
                if self.win_loss < 0:
                    close_bonus = -0.1
                elif self.win_loss == 0:
                    close_bonus = 0
                elif self.win_loss > 0:
                    close_bonus = 0.1
           
            if 0 <= abs_win_loss < 10:
                step_bonus = 0.02 
            elif 10 <= abs_win_loss < 20:
                step_bonus = 0.05 
            elif abs_win_loss >= 20:
                step_bonus = 0.06 

            reward = self.difference * step_bonus + close_bonus

            if close:
                self.prev_win_loss = 0
                self.win_loss = 0
            else:
                self.prev_win_loss = self.win_loss

        return reward


    def epoch_end(self, current_state):
        sample_time = env.to_datetime(current_state)
        if int(sample_time.month) == 12 and int(sample_time.day) == 20:
            day_over = True
        else:
            day_over = False
        return day_over

    #CHECKS WHETHER THE EPISODE IS DONE. EG IS IT A NEW DAY?
    def episode_done(self, current_state):

        current_flat = current_state
        if (current_flat[STACK_SIZE-1,4] != self.prev_day): 
            episode_done = True
        else:
            episode_done = False
        self.prev_day = current_flat[STACK_SIZE-1,4]
        return episode_done


#SUMTREE USED FOR PRIORITEZED EXPERINCE REPLAY. 
class SumTree(object):
    data_pointer = 0
    
    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

#CUSTOM SCALER USED DUE TO MEMORYLEAK IN SKLEARN MINMAX SCALER
class scaler():
    
    def __init__(self):
        
        self.scale = np.empty((2,SCALE_SPACE))
        self.scaled_array = np.empty((STACK_SIZE,ENVIROMENT_OBSERVATION_SPACE), dtype=np.float)
   
    def custom_fit(self, array):

        array = array[:,:38]

        index = 0
        #scale = cp.empty((2,array.shape[1]))
        for column in array.T:
            self.scale[1,index] = np.amin(column, axis=0)
            if ((np.amax(column, axis=0) - np.amin(column, axis=0))) == 0:
                self.scale[0,index] = 0
            else:
                self.scale[0,index] = 2 / (np.amax(column, axis=0) - np.amin(column, axis=0))
            index += 1
        

    def custom_transform(self, array):
        #print(f"1 Array shape: {array.shape}")
        if array.ndim == 1:
            array = array.reshape(1,-1)
        idx = 0

        self.scaled_array = array.copy()
        sel_array = array[:,:38]
        
        for column in sel_array.T:
            self.scaled_array[:,idx] = self.scale[0,idx] * column + (-1) - self.scale[1,idx] * self.scale[0,idx]
            idx += 1

        return self.scaled_array.astype('float64') 

#CLASS FOR THE MODEL
class DQNAgent:
    def __init__(self):
        
        #main model # gets trained every step
        self.model = self.load_or_create_model()
        
        #Target model this is what we .predict against every step
        self.target_model = self.create_model()
        self.xgb_model = joblib.load(xgb_filename)

        self.MEMORY = Memory(REPLAY_MEMORY_SIZE)
        
        self.tensorboard = tensorboard
        #self.checkpoint = cp_callback
        self.target_update_counter = 0
        self.TAU = 0.01
        self.steps = 0
        self.learning_rate_set = learning_rate

    #LEARNING RATE SCHEDULER
    def lr_scheduler(self, epoch):
        if epoch < 3:
            return self.learning_rate_set
        elif 3 <= epoch < 12:
            return self.learning_rate_set/10
        elif 12 <= epoch < 20:
            return self.learning_rate_set/100
        else:
            return self.learning_rate_set/1000
        
    #CHECKS WHETHER TO LOAD A SAVED MODEL
    def load_or_create_model(self):
        if LOAD_MODEL:
            model = tf.keras.models.load_model(MODEL_PATH)
            print('Loaded {}'.format(MODEL_FILENAME))
            return model
        else:
            model = self.create_model()
            print('Created new model!')
        return model
        
    #MODEL. ACTUALLY TWO. ONE VALUE MODEL AND AN ADVANTAGE MODEL AS USED IN DOUBLE DQN MODELS
    def create_model(self):
        
        input_node = tf.keras.Input(shape=(STACK_SIZE, ENVIROMENT_OBSERVATION_SPACE))
        input_layer = input_node

        #define state value function
        
        state_value = tf.keras.layers.LSTM(64, return_sequences=True, stateful=False, activation='tanh')(input_layer)
        state_value = tf.keras.layers.Dropout(0.2)(state_value)
        
        state_value = tf.keras.layers.LSTM(256, return_sequences=True, stateful=False, activation='tanh')(state_value)
        state_value = tf.keras.layers.Dropout(0.2)(state_value)

        state_value = tf.keras.layers.LSTM(192, return_sequences=True, stateful=False, activation='tanh')(state_value)
        state_value = tf.keras.layers.Dropout(0.2)(state_value)

        state_value = tf.keras.layers.LSTM(96, return_sequences=False, stateful=False, activation='tanh')(state_value)
        state_value = tf.keras.layers.Dropout(0.2)(state_value)
       
        state_value = tf.keras.layers.BatchNormalization()(state_value)
        state_value = tf.keras.layers.Dense(240, activation='relu')(state_value)
        

        state_value = tf.keras.layers.Dense(1, activation='linear')(state_value)
        state_value = tf.keras.layers.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], axis=-1), output_shape=(ACTION_SPACE,))(state_value)

        #define acion advantage

        action_advantage = tf.keras.layers.LSTM(64, return_sequences=True, stateful=False, activation='tanh')(input_layer)
        action_advantage = tf.keras.layers.Dropout(0.2)(action_advantage)
        
        action_advantage = tf.keras.layers.LSTM(256, return_sequences=True, stateful=False, activation='tanh')(action_advantage)
        action_advantage = tf.keras.layers.Dropout(0.2)(action_advantage)

        action_advantage = tf.keras.layers.LSTM(192, return_sequences=True, stateful=False, activation='tanh')(action_advantage)
        action_advantage = tf.keras.layers.Dropout(0.2)(action_advantage)

        action_advantage = tf.keras.layers.LSTM(96, return_sequences=False, stateful=False, activation='tanh')(action_advantage)
        action_advantage = tf.keras.layers.Dropout(0.2)(action_advantage)
       
        action_advantage = tf.keras.layers.BatchNormalization()(action_advantage)
        action_advantage = tf.keras.layers.Dense(240, activation='relu')(action_advantage)
        
        action_advantage = tf.keras.layers.Dense(ACTION_SPACE, activation='linear')(action_advantage)
        action_advantage = tf.keras.layers.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True), output_shape=(ACTION_SPACE,))(action_advantage)

        #merge by adding
        Q = tf.keras.layers.add([state_value,action_advantage])

        #define model
        model = tf.keras.Model(inputs=input_node, outputs=Q)
                
        #Model compile settings:
        opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        
        # Compile model
        model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
        )
        return model
    
    def update_replay_memory(self, transition):
        self.MEMORY.store(transition)
    
    #TO UPDATE TARGET MODEL WE USE A SOFT UPDATE METHOD
    def update_target_model(self):
        #Soft update of target model
        q_model_theta = self.model.get_weights()
        target_model_theta = self.target_model.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_model_theta, target_model_theta):
            target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
            target_model_theta[counter] = target_weight
            counter += 1
            self.target_model.set_weights(target_model_theta)
    
    #IF THE MEMORY IS FILLED TO MINIMUM WE START TRAING THE MODEL
    def train(self, step):
        self.steps += 1

        #Tests whether the memory is filled otherwise it return to main loop
        if self.steps < MIN_REPLAY_MEMORY_SIZE:
            return
        if self.steps == MIN_REPLAY_MEMORY_SIZE:
            print(f"-------------------------STEPS: {self.steps}---------------------")

        #Fetch minibatch from memory
        tree_idx, minibatch = self.MEMORY.sample(MINIBATCH_SIZE)
        wrong_shape = True
        count = 0

        #Some error handling needed
        while wrong_shape:
            current_state_mini = np.array([transition[0] for transition in minibatch])
            if current_state_mini.shape != (MINIBATCH_SIZE, STACK_SIZE, ENVIROMENT_OBSERVATION_SPACE):
                print(f'ERROR SHAPE: {current_state_mini.shape}')
                count += 1
                if count > 3:
                    break
            else:
                wrong_shape = False

        current_qs_list = self.model.predict(current_state_mini)
       
        new_state_mini = np.array([transition[3] for transition in minibatch])
        
        future_qs_list = self.target_model.predict(new_state_mini)
        
        new_state_pred = self.model.predict(new_state_mini)

        X = []
        y = []
        
        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            
            if not done:          
                
                a = np.argmax(new_state_pred[index])
                new_q = reward + DISCOUNT * (future_qs_list[index][a])             
            else:
                new_q = reward
   
            # Update Q value for given state
            qs = current_qs_list[index]

            qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(qs)
        
        #PER related
        indices = np.arange(MINIBATCH_SIZE, dtype=np.int32)
       
        
        absolute_errors = np.abs(current_qs_list[indices, np.array(action)]- np.asarray(y)[indices, np.array(action)])
        
        # Update priority experience replay memory
        self.MEMORY.batch_update(tree_idx, absolute_errors)
          
        # Fit on all samples as one batch, log only on terminal state
        history = self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if done else None)

        return history.history['loss'][0]
            
    # Fetches Q values from Value model given current state.
    def get_qs(self, state):
        return self.model.predict(np.expand_dims(myscaler.custom_transform(state), axis=0))
        
#TIMER USED FOR UNIT TESTING, TO OPTIMIZE AND FIND SLOW PERFORMING CODE.
def code_timer(timer_item, start_time):
    sum_time = timeit.default_timer()-start_time
    if timer_item in timer_dict:    
        timer_dict[timer_item] = timer_dict[timer_item] + sum_time
    else:
        timer_dict[timer_item] = sum_time

#HERE STARTS THE MAIN PROGRAM
#MODELS, SCALER, ENVIROMENT ETC. ARE INITIALIZED
timer_dict = {}
total_eps = 1
close_market = False
agent = DQNAgent()
env = forex_env()
myscaler = scaler()
prev_epoch_win_loss = 0

if LOAD_MODEL:
    epsilon = 0.5
    prev_epoch = 33
else:
    prev_epoch = 0
    
for i in range(EPOCHS):
    epoch_start = datetime.now()

    print(f'EPOCH # {i+1} starting, of {EPOCHS} epochs. Start time: {epoch_start.strftime("%Y%m%d_%H%M%S")}')
    epoch_count = i+1 #USED FOR CHECKPOINT CALLBACK IDENTIFICATION
    terminal_state = False
    
    learning_rate = agent.lr_scheduler(i+1+prev_epoch)
    print(f'LEARNING RATE = {learning_rate}')
    current_state = env.preprocess(filename)
    
    step = 1
    
    env.reset()
    
    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):

        start_time = datetime.now()
        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        
        done = False

        while not done:
            done = env.episode_done(current_state)
          
            if terminal_state:
                action = 3
                print('TERMINAL STATE')
            elif i < -3 and not LOAD_MODEL:
                #Get action from pretrained XGBoost model
                action = int(agent.xgb_model.predict(current_state[STACK_SIZE-1:STACK_SIZE,3:-1]))

            elif np.random.random() > epsilon:
                # Get action from Q table      
                action = np.argmax(agent.get_qs(current_state[:,3:]))
                                 
            else:
                # Get random action
                action = np.random.randint(0, 4)
                #print("XGB ACTION!")
            action = env.action_check(current_state, action, close_market)

            (
                new_state,
                terminal_state,
                close,
                step,
            ) = env.to_market(current_state, action, step)

            reward = env.get_reward(close)
            episode_reward += reward

            # Every step we update replay memory and train main network
            scaled_current = myscaler.custom_transform(current_state[:,3:])
            
            scaled_new_state = np.squeeze((myscaler.custom_transform(new_state[:,3:])))
            
            agent.update_replay_memory((scaled_current, action, reward, scaled_new_state, done))

            out_loss = agent.train(step)
            # step += 1
           
            current_state = new_state
            market_close = False
            

        # every step update target model
        agent.update_target_model()
        
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)

        #TENSORBOARD LOGS FOR EACH EPISODE
        with file_writer.as_default():
            tf.summary.scalar("1/Reward", sum(ep_rewards), step= total_eps+episode)
            tf.summary.scalar("1/Pips", env.total_win_loss, step= total_eps+episode)
            if out_loss != None:
                tf.summary.scalar("1/Loss", out_loss, step = total_eps+episode)
        file_writer.flush()

        end_time = datetime.now()
        acc_run_time = end_time - start_time
        
        print(f"Episode took {acc_run_time} to run")

        # DECAY EPSILON
        if epsilon > MIN_EPSILON and i>2: #We only change epsilon after Epoch 1 as we use XGBooster instead of random action
            epsilon *= EPSILON_DECAY
            print(f'EPSILON: {epsilon}')
        
    total_eps += episode
    end_time2 = datetime.now()
    time_now = end_time2.strftime("%Y%m%d_%H%M%S")
    # Save model
    agent.model.save(f"models\\EPOCH{i+1}_{MODEL_NAME}__{SYMBOL}_{MODEL_ID}_EPS{total_eps}.h5")
    checkpoint()
    
    epoch_wl = env.total_win_loss - prev_epoch_win_loss

    #TENSORBOARD LOGS FOR EACH EPOCH
    with file_writer.as_default():
        tf.summary.scalar("2/Epoch Pips", epoch_wl, step= i+1)
        tf.summary.scalar("2/Epsilon", epsilon, step= i+1)
        tf.summary.scalar("2/Learning Rate", learning_rate, step= i+1)
    file_writer.flush()
    prev_epoch_win_loss = env.total_win_loss
    print('---------------------------------------------------------------------')
    print(f"EPOCH # {i+1} ended, of {EPOCHS} epochs. Earned {epoch_wl} pips this epoch.")
    print(f'EPOCH TOOK: {end_time2 - epoch_start} TO FINISH')
    print('---------------------------------------------------------------------')
    

