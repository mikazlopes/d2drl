import gym
from gym import spaces
from gym.wrappers import ResizeObservation
import requests
import time
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import logging
import random
import math
from d2stream import D2GameState  # Import the class from d2stream.py

# Setup logging at the start of your script
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Define the gym environment
class DiabloIIGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_STEPS_NO_REWARD = 1000

    def __init__(self, server_url, flask_port):
        
        self.d2_game_state = D2GameState(flask_port)
        self.d2_game_state.run_server()
        self.cumulative_reward = 0
        self.steps_since_last_reward = 0
        self.step_counter = 0

        self.key_mapping = ['a', 't', 's', 'i', '1', '2', '3', '4', 'r', None]
        self.keyboard_action_space = len(self.key_mapping)

        self.action_space = spaces.Box(low=np.array([5, 30, 0, 0], dtype=np.float32), high=np.array([795, 520, 1, self.keyboard_action_space - 1], dtype=np.float32))
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        #self.observation = np.zeros(self.observation_space.shape, dtype=self.observation_space)

        self.server_url = server_url
    
    def send_request(self, url, data, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=data)
                response.raise_for_status()  # This will raise an exception for HTTP errors
                return response.json().get('success', False)
            except requests.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # If this was the last attempt, re-raise the exception
                    raise
                time.sleep(1)  # Optional: sleep for a bit before retrying

        # If all retries fail, this will never be reached due to the raise in the except block
        return False

    def step(self, action):
        self.step_counter += 1
        
        current_state = self.d2_game_state.get_state()

        # Decode mouse actions from the action vector
        mouse_x, mouse_y, mouse_click_continuous, keyboard_action_encoded = action

        # Convert mouse_click from continuous back to binary
        mouse_click_action = 'left' if mouse_click_continuous > 0.5 else 'right'

        # Decode the keyboard action, ensuring it's within the valid range
        keyboard_action = int(np.round(keyboard_action_encoded))
        keypress_action_key = None
        if 0 <= keyboard_action < len(self.key_mapping):
            keypress_action_key = self.key_mapping[keyboard_action]

        # Prepare the combined mouse action
        combined_mouse_action = {
            "x": int(mouse_x),
            "y": int(mouse_y),
        }

        # Prepare the mouse click action separately
        mouse_click_action_dict = {
            "button": mouse_click_action
        }

        # Prepare the combined action dictionary
        combined_action = {
            "mouse_move_action": combined_mouse_action,
            "mouse_click_action": mouse_click_action_dict  # Now sending mouse click action separately
        }

        if keypress_action_key:
            combined_action['keypress_action'] = {"key": keypress_action_key}

        # Send the combined request
        if not self.send_request(f"{self.server_url}/combined_action", combined_action):
            print("Combined action failed")

        # Get a screenshot for the observation
        response = requests.get(f"{self.server_url}/screenshotsmall")
        image = Image.open(BytesIO(response.content))
        observation_image = np.array(image)
        # Ensure the observation matches the expected shape (64, 64, 3)
        if observation_image.shape != (64, 64, 3):
            # Resize and/or adjust observation as needed
            observation_image = cv2.resize(observation_image, (64, 64))
            if observation_image.ndim == 2:  # If grayscale, convert to 3 channels
                observation_image = np.stack((observation_image,)*3, axis=-1)

        # Ensure observation is of type uint8
        observation_image = np.clip(observation_image, 0, 255).astype(np.uint8)

        assert observation_image.shape == (64, 64, 3), "Observation shape mismatch"

        observation = observation_image

        updated_state = self.d2_game_state.get_state() 
        
        done = updated_state.get('IsDead', False)
        
        reward = self.calculate_reward(new_state=updated_state, old_state=current_state)
        if reward <= 0:
            self.steps_since_last_reward += 1
        else:
            self.steps_since_last_reward = 0
        self.cumulative_reward += reward

        if self.steps_since_last_reward >= self.MAX_STEPS_NO_REWARD or done:
            done = True

        self.d2_game_state.previous_quests = updated_state['CompletedQuests'].copy()
        self.d2_game_state.game_state['RemovedItems'] = []
        
        info = {}

        logging.info(f"Step: {self.steps_since_last_reward}, Reward: {reward}, Cumulative Reward: {self.cumulative_reward}, Done: {done}")

        return observation, reward, done, info

    
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
            'Strength': 2,
            'Dexterity': 1,
            'ManaMax': 1,
            'LifeMax': 3
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
            reward += 500  # Assign the new area discovery reward
            self.d2_game_state.game_state['NewAreaDiscovered'] = False

        # Quest completion rewards
        new_quests = new_state['CompletedQuests']
        old_quests = self.d2_game_state.previous_quests
        for difficulty in new_quests.keys():
            new_quests_diff = set(new_quests[difficulty]) - set(old_quests.get(difficulty, []))
            if difficulty == 'Normal':
                reward += 50000 * len(new_quests_diff)
            elif difficulty == 'Nightmare':
                reward += 70000 * len(new_quests_diff)
            elif difficulty == 'Hell':
                reward += 100000 * len(new_quests_diff)

        # Reward for achieving quest milestones
        if 'QuestPartsCompleted' in new_state and 'QuestPartsCompleted' in old_state:
            if new_state['QuestPartsCompleted'] > old_state['QuestPartsCompleted']:
                reward += 10000 * (new_state['QuestPartsCompleted'] - old_state['QuestPartsCompleted'])

        return reward

    def reset_sequence_dead(self):
        
        template_path = 'template_save.png'  # Provide the correct path to your template image
        template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
        self.send_keypress('esc')

        menu_matched = False
        while not menu_matched:
            # Request a screenshot
            response = requests.get(f"{self.server_url}/screenshotreset")
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

    def template_match(self, screenshot_np, template_np, threshold=0.3):
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
        attempt_count = 0
        max_attempts = 3

        while not menu_matched:
            # Request a screenshot
            response = requests.get(f"{self.server_url}/screenshotreset")
            screenshot = Image.open(BytesIO(response.content))
            screenshot_np = np.array(screenshot)

            # Check if the menu text is visible in the screenshot
            menu_matched = self.template_match(screenshot_np, template_img)
            
            if not menu_matched:
                attempt_count += 1
                if attempt_count % max_attempts == 0:
                
                    #move char to different location
                    numbers = [20,500]
                    chosen_number = random.choice(numbers)
                    self.send_mouse_move(400, chosen_number)
                    self.send_mouse_click('left')
                    time.sleep(3)  # Add delay after mouse move and click

                # Send ESC keypress and try again
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

       # Get a screenshot for the observation
        response = requests.get(f"{self.server_url}/screenshotsmall")
        image = Image.open(BytesIO(response.content))
 
        observation_image = np.array(image)
        # Ensure the observation matches the expected shape (64, 64, 3)
        if observation_image.shape != (64, 64, 3):
            # Resize and/or adjust observation as needed
            observation_image = cv2.resize(observation_image, (64, 64))
            if observation_image.ndim == 2:  # If grayscale, convert to 3 channels
                observation_image = np.stack((observation_image,)*3, axis=-1)

        # Ensure observation is of type uint8
        observation_image = np.clip(observation_image, 0, 255).astype(np.uint8)

        assert observation_image.shape == (64, 64, 3), "Observation shape mismatch"

        observation = observation_image
        

        # Reset Areas and other variables

        self.d2_game_state.game_state['Areas'] = []
        
        return observation, {}


    def render(self, mode='human', close=False):
        # Rendering not required for this setup
        pass


