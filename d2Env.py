import gymnasium as gym
from gymnasium import spaces
import requests
import time
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import logging
from collections import deque
from d2stream import D2GameState  # Import the class from d2stream.py

# Setup logging at the start of your script
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


# Define the gym environment
class DiabloIIGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_STEPS_NO_REWARD = 1000  # Max steps without a reward before resetting


    def __init__(self, server_url, flask_port):
        super(DiabloIIGymEnv, self).__init__()
        
        self.d2_game_state = D2GameState(flask_port)
        self.d2_game_state.run_server()
        self.cumulative_reward = 0
        self.steps_since_last_reward = 0
        self.step_counter = 0
        self.frame_stack = deque(maxlen=4)  # Stores the last 4 frames

        # Add a None value to represent no keystroke
        self.key_mapping = ['a', 't', 's', 'i', '1', '2', '3', '4', 'r', 'Alt', 'Tab', None]

        # Now the action space for the keypress_index has to be one more than the length of self.key_mapping
        self.action_space = spaces.MultiDiscrete([791, 551, 2, len(self.key_mapping)])

        # Example observation space, which will be an image buffer mixed with data
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(12, 300, 400), dtype=np.uint8),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)  # Example shape (10,)
        })

        # Initialize frame stack with four black frames
        initial_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        for _ in range(4):
            self.frame_stack.append(initial_frame)


        # Flask server URL
        self.server_url = server_url
    
    def send_request(self, url, data):
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # This will raise an exception for HTTP errors
            return response.json().get('success', False)
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return False

    def step(self, action):

        self.step_counter += 1
        
        current_state = self.d2_game_state.get_state()
        
        # Extract the discrete actions from the MultiDiscrete space
        mouse_x, mouse_y, mouse_click, keypress_index = action

        # Convert NumPy int64 types to native Python int using .item()
        mouse_x_action = int(10 + mouse_x.item())
        mouse_y_action = int(30 + mouse_y.item())
        mouse_click_action = 'left' if mouse_click.item() == 0 else 'right'
        
        # Handle the action for keypress
        keypress_index = action[3]  # This gets the index for the keypress action
        keypress_action_key = self.key_mapping[keypress_index]  # This can now be None

        if keypress_action_key is not None:
            # Perform the key press
            keypress_action = {
                "key": keypress_action_key,
                "action": "press"
            }
            if not self.send_request(f"{self.server_url}/keypress", keypress_action):
                print("Keypress action failed, skipping to the next action")

        # Now create the JSON payloads with native Python types for mouse actions
        mouse_action = {
            "action": "move",
            "x": mouse_x_action,
            "y": mouse_y_action
        }
        if not self.send_request(f"{self.server_url}/mouse", mouse_action):
            print("Mouse move action failed, skipping to the next action")

        click_action = {
            "action": "click",
            "button": mouse_click_action
        }
        if not self.send_request(f"{self.server_url}/mouse", click_action):
            print("Mouse click action failed, skipping to the next action")


        # Get a screenshot for the observation and append it to the frame stack
        response = requests.get(f"{self.server_url}/screenshot")
        image = Image.open(BytesIO(response.content))
        frame = np.array(image)

        # Replace the oldest frame
        self.frame_stack.append(frame)
        if len(self.frame_stack) > 4:
            self.frame_stack.popleft()

        # Check and stack frames
        if all(frame.shape == (300, 400, 3) for frame in self.frame_stack):
            stacked_frames = np.concatenate([frame for frame in self.frame_stack], axis=2)  # Shape should be (300, 400, 12)
            stacked_frames = np.transpose(stacked_frames, (2, 0, 1))  # Transpose to (12, 300, 400)
        else:
            raise ValueError("Step: Frame shapes do not match expected dimensions.")

        
        # Get the updated game state after the action
        updated_state = self.d2_game_state.get_state() 

        # Get the scalar values from the game state
        vector_obs = np.array([
            updated_state.get('Life', 0),
            updated_state.get('LifeMax', 0),
            updated_state.get('Mana', 0),
            updated_state.get('ManaMax', 0),
            updated_state.get('Level', 0),
            updated_state.get('Experience', 0),
            updated_state.get('Skills', 0),
            updated_state.get('Strength', 0),
            updated_state.get('Dexterity', 0),
            updated_state.get('KilledMonsters', 0),
            updated_state.get('FireResist', 0),
            updated_state.get('ColdResist', 0),
            updated_state.get('LightningResist', 0),
            updated_state.get('PoisonResist', 0),
            updated_state.get('MagicFind', 0),
            updated_state.get('FasterCastRate', 0),
            updated_state.get('IncreasedAttackSpeed', 0),
            updated_state.get('FasterRunWalk', 0),
            updated_state.get('FasterHitRecovery', 0),
            updated_state.get('Gold', 0),
            updated_state.get('GoldStash', 0),
            updated_state.get('Area', 1),
            updated_state.get('DamageMax', 0),
            updated_state.get('Defense', 0),
            updated_state.get('AttackRating', 0),
        ])

        observation = {
        "image": stacked_frames,  # The image from the screenshot
        "vector": vector_obs,         # The vector of scalar values
        }

        # Check if the hero is dead to decide if the episode should be done
        done = updated_state.get('IsDead', False)
        
        # Calculate the reward based on the changes in state
        reward = self.calculate_reward(new_state=updated_state, old_state=current_state)

        if reward <= 0:
            self.steps_since_last_reward += 1
        else:
            self.steps_since_last_reward = 0
        
        self.cumulative_reward += reward  # Add the received reward to the cumulative reward


        # Check if the reset is needed based on the custom logic
        if self.steps_since_last_reward >= self.MAX_STEPS_NO_REWARD:
            done = True
            # You can also include any additional logic here if needed before the reset
        else:
            done = updated_state.get('IsDead', False)
        
        # Update the old state with the current state for the next step
        self.d2_game_state.previous_quests = updated_state['CompletedQuests'].copy()

        # Clear RemovedItems after the reward calculation
        self.d2_game_state.game_state['RemovedItems'] = []
        
        info = {}  # Additional info, if any, for debugging purposes
        truncated = False  # This environment does not use truncation

        logging.info(f"Step: {self.steps_since_last_reward}, Action: {action}, Reward: {reward}, Sum Reward: {self.cumulative_reward}, Done: {done}")

        return observation, reward, done, truncated, info
    
    def calculate_potion_penalty_reward(self, new_state, old_state):
        reward = 0
        life_max = new_state.get('LifeMax', 0)
        mana_max = new_state.get('ManaMax', 0)
        current_life = new_state.get('Life', 0)
        current_mana = new_state.get('Mana', 0)
        removed_items = new_state.get('RemovedItems', [])

        for item in removed_items:
        
            item_base_name = item.get('ItemName', '')  # Changed from ItemBaseName to ItemName to match your JSON structure
            if 'Healing Potion' in item_base_name:
                if current_life == life_max:
                    logging.info(f"Potion penalty applied: -2 (Life: {current_life}, LifeMax: {life_max})")
                    reward -= 5
                elif current_life <= life_max * 0.5:
                    logging.info(f"Potion reward applied: +2 (Life: {current_life}, LifeMax: {life_max})")
                    reward += 10
            elif 'Mana Potion' in item_base_name:
                if current_mana == mana_max:
                    logging.info(f"Potion penalty applied: -2 (Mana: {current_mana}, ManaMax: {mana_max})")
                    reward -= 5
                elif current_mana <= mana_max * 0.5:
                    logging.info(f"Potion reward applied: +2 (Mana: {current_mana}, ManaMax: {mana_max})")
                    reward += 10

        return reward

    def calculate_reward(self, new_state, old_state):
        reward = 0

        # Calculate reward based on the change of attributes
        attribute_rewards = {
            'Strength': 5,
            'Dexterity': 1,
            'ManaMax': 1,
            'LifeMax': 2
        }

        # Calculate potion penalties or rewards
        potion_reward = self.calculate_potion_penalty_reward(new_state, old_state)
        reward += potion_reward

        # Attribute-based rewards
        for attr, multiplier in attribute_rewards.items():
            if new_state.get(attr, 0) > old_state.get(attr, 0):
                reward += (new_state[attr] - old_state[attr]) * multiplier

        # Reward for life and mana increase
        for attr in ['Life', 'Mana']:
            if new_state.get(attr, 0) > old_state.get(attr, 0):
                reward += (new_state[attr] - old_state[attr]) * 0.5

        # Reward or penalty for DamageMax and Defense
        if new_state.get('DamageMax', 0) > old_state.get('DamageMax', 0):
            reward += 20
        elif new_state.get('DamageMax', 0) < old_state.get('DamageMax', 0):
            reward -= 20

        if new_state.get('Defense', 0) > old_state.get('Defense', 0):
            reward += 10
        elif new_state.get('Defense', 0) < old_state.get('Defense', 0):
            reward -= 10

        # Reward for killing monsters
        if new_state.get('KilledMonsters', 0) > old_state.get('KilledMonsters', 0):
            reward += 20

        # Reward for gold
        if new_state.get('Gold', 0) > old_state.get('Gold', 0):
            reward += 2
        if new_state.get('GoldStash', 0) > old_state.get('GoldStash', 0):
            reward += 5

        # Reward for fcr, fhr, frw, ias, and mf
        for attr in ['FasterCastRate', 'FasterHitRecovery', 'FasterRunWalk', 'IncreasedAttackSpeed', 'MagicFind']:
            if new_state.get(attr, 0) > old_state.get(attr, 0):
                reward += new_state[attr] - old_state[attr]

        # Reward for resistances
        for res in ['FireResist', 'ColdResist', 'LightningResist', 'PoisonResist']:
            if new_state.get(res, 0) > old_state.get(res, 0):
                reward += new_state[res] - old_state[res]

        # Reward for level up
        if new_state.get('Level', 0) > old_state.get('Level', 0):
            reward += 100

        # Penalty for death
        if new_state.get('IsDead', False):
            reward -= 500

        # Reward for experience gain
        if new_state.get('Experience', 0) > old_state.get('Experience', 0):
            reward += 20

        # Check for new area discovery
        if new_state.get('NewAreaDiscovered', False):
            reward += 700  # Assign the new area discovery reward
            self.d2_game_state.game_state['NewAreaDiscovered'] = False

        # Quest completion rewards
        new_quests = new_state['CompletedQuests']
        old_quests = self.d2_game_state.previous_quests
        for difficulty in new_quests.keys():
            new_quests_diff = set(new_quests[difficulty]) - set(old_quests.get(difficulty, []))
            if difficulty == 'Normal':
                reward += 5000 * len(new_quests_diff)
            elif difficulty == 'Nightmare':
                reward += 7000 * len(new_quests_diff)
            elif difficulty == 'Hell':
                reward += 10000 * len(new_quests_diff)

        # Reward for achieving quest milestones
        if 'QuestPartsCompleted' in new_state and 'QuestPartsCompleted' in old_state:
            if new_state['QuestPartsCompleted'] > old_state['QuestPartsCompleted']:
                reward += 1000 * (new_state['QuestPartsCompleted'] - old_state['QuestPartsCompleted'])

        return reward

    def reset_sequence_dead(self):
        
        self.send_keypress('esc')

        template_path = 'template_image.png'  # Provide the correct path to your template image
        template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)

        menu_matched = False
        while not menu_matched:
            # Request a screenshot
            response = requests.get(f"{self.server_url}/screenshot400")
            screenshot = Image.open(BytesIO(response.content))
            screenshot_np = np.array(screenshot)

            # Check if the menu text is visible in the screenshot
            menu_matched = self.template_match(screenshot_np, template_img)
            
            if not menu_matched:
                # If not visible, send ESC and try again
                self.send_keypress('esc')
                time.sleep(1)  # Add a delay to allow for the menu to appear

        self.send_mouse_move(400, 280)
        self.send_mouse_click('left')
        self.send_mouse_move(400, 325)
        self.send_mouse_click('left')
        self.send_mouse_move(550, 525)
        self.send_mouse_click('left')
        self.send_mouse_move(490, 345)
        self.send_mouse_click('left')
        self.send_mouse_move(100, 525)
        self.send_mouse_click('left')
        self.send_mouse_move(550, 325)
        self.send_mouse_click('left')
        self.send_keypress('a')
        self.send_keypress('b')
        self.send_keypress('Enter')
        time.sleep(2)

    def template_match(self, screenshot_np, template_np, threshold=0.35):
        # If the template image is not grayscale, convert it
        if len(template_np.shape) == 3:
            template_gray = cv2.cvtColor(template_np, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_np

        # No need to convert screenshot as it is already in grayscale
        screenshot_gray = screenshot_np

        # Perform template matching
        res = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            # Template matched successfully
            print(max_val)
            return True
        else:
            # Template did not match
            print(max_val)
            return False

        
    def reset_sequence_noreward(self):
        template_path = 'template_save.png'  # Provide the correct path to your template image
        template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)

        menu_matched = False
        while not menu_matched:
            # Request a screenshot
            response = requests.get(f"{self.server_url}/screenshot400")
            screenshot = Image.open(BytesIO(response.content))
            screenshot_np = np.array(screenshot)

            # Check if the menu text is visible in the screenshot
            menu_matched = self.template_match(screenshot_np, template_img)
            
            if not menu_matched:
                # If not visible, send ESC and try again
                self.send_keypress('esc')
                time.sleep(1)  # Add a delay to allow for the menu to appear

        # Proceed with the reset sequence since the menu text is visible
        self.send_mouse_move(400, 280)
        self.send_mouse_click('left')
        self.send_mouse_move(400, 325)
        self.send_mouse_click('left')
        self.send_mouse_move(550, 525)
        self.send_mouse_click('left')
        self.send_mouse_move(490, 345)
        self.send_mouse_click('left')
        self.send_mouse_move(100, 525)
        self.send_mouse_click('left')
        self.send_mouse_move(550, 325)
        self.send_mouse_click('left')
        self.send_keypress('a')
        self.send_keypress('b')
        self.send_keypress('Enter')
        time.sleep(2)


    def send_keypress(self, key):
        keypress_action = {"key": key, "action": "press"}
        requests.post(f"{self.server_url}/keypress", json=keypress_action)

    def send_mouse_move(self, x, y):
        mouse_action = {"action": "move", "x": x, "y": y}
        requests.post(f"{self.server_url}/mouse", json=mouse_action)

    def send_mouse_click(self, button):
        click_action = {"action": "click", "button": button}
        requests.post(f"{self.server_url}/mouse", json=click_action)

    def reset(self, **kwargs):

        if not hasattr(self, 'initial_reset_done'):
            # Perform actions for the very first reset
            self.send_mouse_move(400, 325)
            self.send_mouse_click('left')
            self.send_mouse_move(550, 325)
            self.send_mouse_click('left')
            self.send_keypress('a')
            self.send_keypress('b')
            self.send_keypress('Enter')
            time.sleep(4)
            # Set the flag indicating that initial reset actions have been performed
            self.initial_reset_done = True

        # Check for the custom reset condition
        if self.steps_since_last_reward >= self.MAX_STEPS_NO_REWARD:
            self.reset_sequence_noreward()
            self.cumulative_reward = 0
            self.steps_since_last_reward = 0
        elif self.d2_game_state.game_state.get('IsDead', False):
            self.reset_sequence_dead()
            self.cumulative_reward = 0
            self.steps_since_last_reward = 0
        
        # Reset Areas and other variables

        self.d2_game_state.game_state['Areas'] = []

       # Clear the frame stack
        self.frame_stack.clear()

        # Fetch the initial screenshot
        response = requests.get(f"{self.server_url}/screenshot")
        initial_image = Image.open(BytesIO(response.content))
        initial_frame = np.array(initial_image)

        # Check and fill the frame stack
        if initial_frame.shape == (300, 400, 3):
            for _ in range(4):
                self.frame_stack.append(initial_frame)

            stacked_frames = np.concatenate([frame for frame in self.frame_stack], axis=2)  # Shape should be (300, 400, 12)
            stacked_frames = np.transpose(stacked_frames, (2, 0, 1))  # Transpose to (12, 300, 400)
        else:
            raise ValueError("Initial frame shape does not match expected dimensions.")
        
        # Get the updated game state after the action
        updated_state = self.d2_game_state.get_state() 

        # Get the scalar values from the game state
        vector_obs = np.array([
            updated_state.get('Life', 0),
            updated_state.get('LifeMax', 0),
            updated_state.get('Mana', 0),
            updated_state.get('ManaMax', 0),
            updated_state.get('Level', 0),
            updated_state.get('Experience', 0),
            updated_state.get('Skills', 0),
            updated_state.get('Strength', 0),
            updated_state.get('Dexterity', 0),
            updated_state.get('KilledMonsters', 0),
            updated_state.get('FireResist', 0),
            updated_state.get('ColdResist', 0),
            updated_state.get('LightningResist', 0),
            updated_state.get('PoisonResist', 0),
            updated_state.get('MagicFind', 0),
            updated_state.get('FasterCastRate', 0),
            updated_state.get('IncreasedAttackSpeed', 0),
            updated_state.get('FasterRunWalk', 0),
            updated_state.get('FasterHitRecovery', 0),
            updated_state.get('Gold', 0),
            updated_state.get('GoldStash', 0),
            updated_state.get('Area', 1),
            updated_state.get('DamageMax', 0),
            updated_state.get('Defense', 0),
            updated_state.get('AttackRating', 0),
        ])

        observation = {
        "image": stacked_frames,  # The image from the screenshot
        "vector": vector_obs,         # The vector of scalar values
        }

        # Reset Areas and other variables

        self.d2_game_state.game_state['Areas'] = []
        
        return observation, {}


    def render(self, mode='human', close=False):
        # Rendering not required for this setup
        pass


