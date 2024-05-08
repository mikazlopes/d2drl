import gymnasium as gym
from gymnasium import spaces
from collections import deque
import requests
import os
import csv
from pathlib import Path
import time
from datetime import datetime
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
import cv2
import logging
import random
from d2stream import D2GameState  # Import the class from d2stream.py

# Setup logging at the start of your script
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


# Define the gym environment
class DiabloIIGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_STEPS_NO_REWARD = 1000  # Max steps without a reward before resetting


    def __init__(self, server_url, flask_port, env_id='main'):
        super(DiabloIIGymEnv, self).__init__()
        
        self.d2_game_state = D2GameState(flask_port)
        self.d2_game_state.run_server()
        self.cumulative_reward = 0
        self.steps_since_last_reward = 0
        self.step_counter = 0
        self.env_id = env_id

        # Initialize episode counter from the file if it exists
        self.episode_tracker_file = 'episode_tracker_sb3_cnn.csv'
        self.episode_counter, self.best_cumulative_reward = self.load_episode_data()

        # Record break through episodes
        self.video_writer = None
        self.video_buffer = []
        self.video_directory = "./videos/"  # Specify your directory here
        os.makedirs(self.video_directory, exist_ok=True)  # Ensure the directory exists


        self.key_mapping = ['a', 't', 's', 'i', '1', '2', '3', '4', 'r', 'Alt', None]
        self.keyboard_action_space = len(self.key_mapping)
        self.alt_counter = 0
        self.alt_pressed = False

        # Check if windows are open
        self.inventory_open = False
        self.char_open = False
        self.skill_tree_open = False

        # Now the action space for the keypress_index has to be one more than the length of self.key_mapping
        self.action_space = spaces.MultiDiscrete([791, 510, 3, len(self.key_mapping)])

        self.observation_width = 200
        self.observation_height = 150
        self.number_frames = 4
        self.frame_stack = deque(maxlen=self.number_frames)

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3 * self.number_frames), dtype=np.uint8)

        # Flask server URL
        self.server_url = server_url

    def load_episode_data(self):
        episode_counter = 0
        best_cumulative_reward = -float('inf')
        if Path(self.episode_tracker_file).is_file():
            with open(self.episode_tracker_file, mode='r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    if row['env_id'] == self.env_id:
                        episode_counter = int(row['episode_number'])
                        best_cumulative_reward = float(row.get('best_score', -float('inf')))
                        break
        return episode_counter, best_cumulative_reward

    def save_episode_data(self):
        # Load existing data
        data = []
        if Path(self.episode_tracker_file).is_file():
            with open(self.episode_tracker_file, mode='r') as file:
                data = list(csv.DictReader(file))

        # Update or add the episode counter and best score for the current env_id
        updated = False
        for row in data:
            if row['env_id'] == self.env_id:
                row['episode_number'] = str(self.episode_counter)
                row['best_score'] = str(self.best_cumulative_reward)
                updated = True
                break
        if not updated:
            data.append({
                'env_id': self.env_id, 
                'episode_number': str(self.episode_counter),
                'best_score': str(self.best_cumulative_reward)
            })

        # Write updated data back to the file
        with open(self.episode_tracker_file, mode='w', newline='') as file:
            fieldnames = ['env_id', 'episode_number', 'best_score']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    
    def process_image_for_observation(self, pil_image, method='step'):
            
            final_image = pil_image.resize((self.observation_width, self.observation_height), Image.Resampling.LANCZOS)

            # Convert the PIL Image to a NumPy array
            np_image = np.array(final_image)

            # Ensure the observation is of type uint8
            np_image = np.clip(np_image, 0, 255).astype(np.uint8)

            # Ensure the observation matches the expected shape 
            assert np_image.shape == (self.observation_height, self.observation_width, 3), "Observation shape mismatch"

            if method == 'reset':
                for _ in range(self.number_frames):
                    self.frame_stack.append(np_image)
            else:
                self.frame_stack.append(np_image)

            stacked_obs = np.concatenate(list(self.frame_stack), axis=-1)
            
            return stacked_obs
    
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

        # Extract the discrete actions from the MultiDiscrete space
        mouse_x, mouse_y, mouse_click, keypress_index = action

        # ensure X pos is centered
        mouse_x = mouse_x + 5

        # Convert NumPy int64 types to native Python int using .item()
        mouse_x_action = int(10 + mouse_x.item())
        mouse_y_action = int(30 + mouse_y.item())
        if mouse_click.item() == 0:
            mouse_click_action = 'left'
        elif mouse_click.item() == 1:
            mouse_click_action = 'right'
        elif mouse_click.item() == 2:
            mouse_click_action = 'none'

        # Prepare the combined action
        combined_action = {
            'mouse_move_action': {
                "x": mouse_x_action,
                "y": mouse_y_action
            },
            'mouse_click_action': {
                "button": mouse_click_action
            }
        }

        # Add keypress action if applicable and check if screens are open
        keypress_index = action[3]
        keypress_action_key = self.key_mapping[keypress_index]
        
        if keypress_action_key == 'a' and self.char_open == False:
            self.char_open = True
        elif keypress_action_key == 'a' and self.char_open == True:
            self.char_open = False

        if keypress_action_key == 't' and self.skill_tree_open == False:
            self.skill_tree_open = True
            self.inventory_open = False
        elif keypress_action_key == 't' and self.skill_tree_open == True:
            self.skill_tree_open = False
        elif keypress_action_key == 'i' and self.inventory_open == False:
            self.skill_tree_open = False
            self.inventory_open = True
        elif keypress_action_key == 'i' and self.inventory_open == True:
            self.inventory_open = False

        if keypress_action_key == 's':
            self.char_open = False
            self.inventory_open = False
            self.skill_tree_open = False

        if keypress_action_key is not None:
            if keypress_action_key != 'Alt':
                combined_action['keypress_action'] = {
                    "key": keypress_action_key
                }
            else:
                # Logic for keeping Alt pressed for at least 3 steps
                if self.alt_pressed:
                    self.alt_counter = 0
                self.alt_pressed = True
                self.alt_counter += 1
                combined_action['keypress_action'] = {
                    "key": keypress_action_key,
                    "alt_counter": self.alt_counter,
                }
                if self.alt_counter == 3:
                    self.alt_counter = 0
                    self.alt_pressed = False

        # Send the combined request
        if not self.send_request(f"{self.server_url}/combined_action", combined_action):
            print("Combined action failed")


        # Get a screenshot for the observation
        response = requests.get(f"{self.server_url}/screenshot")
        image = Image.open(BytesIO(response.content))
        
        # Append the original image to the video buffer for video saving
        self.video_buffer.append(image)
        # Process the image for the observation
        observation_image = self.process_image_for_observation(image.copy(), 'step')

        observation = observation_image

        #time.sleep(0.3)

        # Get the updated game state after the action
        updated_state = self.d2_game_state.get_state()

        if current_state == updated_state:
            updated_state = self.d2_game_state.get_state()
            # timeout = 0
            # while current_state == updated_state and timeout < 2:
            #     updated_state = self.d2_game_state.get_state()
            #     timeout += 1
            #     time.sleep(0.1)

        # Check if the hero is dead to decide if the episode should be done
        done = updated_state.get('IsDead', False)
        
        # Calculate the reward based on the changes in state
        reward = self.calculate_reward(new_state=updated_state, old_state=current_state)

        if reward <= 0:
            self.steps_since_last_reward += 1
        else:
            self.steps_since_last_reward = 0
        
        PENALTY = -100  # Define the penalty value
        PENALTY_FREQUENCY = 200  # Apply the penalty every 100 steps

        if self.steps_since_last_reward > 0 and self.steps_since_last_reward % PENALTY_FREQUENCY == 0:
            reward += PENALTY  # Notice that we add because reward is typically a negative value

        
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

        if self.char_open and (self.inventory_open or self.skill_tree_open):
            logging.info(f'Env ID: {self.env_id}, Char Screen: {self.char_open}, Inventory Scren: {self.inventory_open}, Skill Screen: {self.skill_tree_open}')

        logging.info(f"Step: {self.steps_since_last_reward}, Env ID: {self.env_id}, Action: {action}, Reward: {reward:.2f}, Sum Reward: {self.cumulative_reward:.2f}, Done: {done}")

        if done:
            self.episode_counter += 1
            logging.info(f'Episode {self.episode_counter} Done')
            self.check_and_save_video()
            self.video_buffer = []  # Reset the buffer for the next episode

        reward = reward / 1e3 # Reward scaling

        return observation, reward, done, truncated, info
    
    def check_and_save_video(self):
        if self.cumulative_reward > self.best_cumulative_reward:
            self.best_cumulative_reward = self.cumulative_reward
            logging.info('Saving Episode Video with new best score: {}'.format(self.best_cumulative_reward))
            self.save_video()
        self.save_episode_data()  # Save the updated episode counter and best score
            
    
    def save_video(self):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_name = f"{self.video_directory}/episode_{self.episode_counter}_{int(self.cumulative_reward)}_{self.env_id}_{current_datetime}.mp4"
        
        # Log the length of the video buffer
        logging.info(f'Number of frames to save: {len(self.video_buffer)}')

        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Use H.264 codec
        out = cv2.VideoWriter(video_name, fourcc, 5.0, (800, 600), True)

        if not out.isOpened():
            logging.error('VideoWriter failed to open')
            return

        try:
            for pil_image in self.video_buffer:
                np_frame = np.array(pil_image)  # Ensure the image is RGB
                bgr_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
                out.write(bgr_frame)  # Write the BGR frame to the video
        except Exception as e:
            logging.error(f'Failed to write frame: {e}')
        finally:
            out.release()
            logging.info(f'Episode video saved at {video_name}')

    
    def calculate_potion_penalty_reward(self, new_state):
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
                    reward += 20
            elif 'Mana Potion' in item_base_name:
                if current_mana == mana_max:
                    logging.info(f"Potion penalty applied: -2 (Mana: {current_mana}, ManaMax: {mana_max})")
                    reward -= 5
                elif current_mana <= mana_max * 0.5:
                    logging.info(f"Potion reward applied: +2 (Mana: {current_mana}, ManaMax: {mana_max})")
                    reward += 20

        return reward

    def calculate_reward(self, new_state, old_state):
        reward = 0

        #Check for open screens

        if self.char_open:
            reward -= 1
        
        if self.inventory_open or self.skill_tree_open:
            reward -= 1

        # Calculate reward based on the change of attributes
        attribute_rewards = {
            'Strength': 5,
            'Dexterity': 3,
            'ManaMax': 1,
            'LifeMax': 8
        }

        # Calculate potion penalties or rewards
        potion_reward = self.calculate_potion_penalty_reward(new_state)
        reward += potion_reward

        # Attribute-based rewards
        for attr, multiplier in attribute_rewards.items():
            if new_state.get(attr, 0) > old_state.get(attr, 0):
                reward += (new_state[attr] - old_state[attr]) * multiplier

        # Reward for life and mana increase
        for attr in ['Life', 'Mana']:
            if new_state.get(attr, 0) > old_state.get(attr, 0):
                reward += (new_state[attr] - old_state[attr]) * 0.1

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
            reward += 400

        # Reward for gold
        if new_state.get('Gold', 0) > old_state.get('Gold', 0):
            reward += 5
        if new_state.get('GoldStash', 0) > old_state.get('GoldStash', 0):
            reward += 10

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
            reward += 500

        # Penalty for death
        if new_state.get('IsDead', False):
            reward -= 500

        # Reward for experience gain
        # if new_state.get('Experience', 0) > old_state.get('Experience', 0):
        #     reward += 200

        # Check for new area discovery
        if new_state.get('NewAreaDiscovered', False):
            reward += 800  # Assign the new area discovery reward
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
        
        # Reward for allocating Skill points
        if 'Skills' in new_state and 'Skills' in old_state:
            if new_state['Skills'] > old_state['Skills']:
                reward += 500 * (new_state['Skills'] - old_state['Skills'])

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
        self.send_mouse_move(622, 325)
        self.send_mouse_click('left')
        self.send_keypress('d')
        self.send_keypress('d')
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
        self.send_mouse_move(622, 325)
        self.send_mouse_click('left')
        self.send_keypress('n')
        self.send_keypress('r')
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
            self.send_mouse_move(622, 325)
            self.send_mouse_click('left')
            self.send_keypress('n')
            self.send_keypress('c')
            self.send_keypress('Enter')
            time.sleep(4)
            # Set the flag indicating that initial reset actions have been performed
            self.initial_reset_done = True

        # Reset Alt
        self.alt_pressed = False

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
        self.skill_tree_open = False
        self.inventory_open = False
        self.char_open = False
        self.d2_game_state.game_state['Areas'] = []

         # Get a screenshot for the observation
        response = requests.get(f"{self.server_url}/screenshot")
        image = Image.open(BytesIO(response.content))

        # Process the image for the observation
        observation_image = self.process_image_for_observation(image.copy(), 'reset')

        observation = observation_image

        # Reset Areas and other variables

        self.d2_game_state.game_state['Areas'] = []

        logging.info(f'{self.env_id} starting Episode {self.episode_counter}')
        
        return observation, {}


    def render(self, mode='human', close=False):
        # Rendering not required for this setup
        pass


