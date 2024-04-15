from flask import Flask, request, jsonify
import threading

class D2GameState:

    def __init__(self, flask_port):
        self.app = Flask(__name__)
        self.game_state = {
            'Event': None,
            'Headers': {},
            'Skills': 0,
            'CompletedQuests': {},
            'AllQuestsDetails': {},
            'QuestPartsCompleted': 0,
            'D2ProcessInfo': {},
            'DIApplicationInfo': {},
            'Level': 0,
            'IsDead': False,
            'Life': 0,
            'LifeMax': 0,
            'Vitality': 0,
            'Dexterity': 0,
            'Mana': 0,
            'ManaMax': 0,
            'Strength': 0,
            'Experience': 0,
            'KilledMonsters': 0,
            'FireResist': 0,
            'ColdResist': 0,
            'LightningResist': 0,
            'PoisonResist': 0,
            'hireling': {},
            'Gold': 0,
            'GoldStash': 0,
            'FasterCastRate': 0,
            'FasterHitRecovery': 0,
            'FasterRunWalk': 0,
            'IncreasedAttackSpeed': 0,
            'MagicFind': 0,
            'Areas': [],
            'Area': 1,
            'DamageMax': 0,
            'Defense': 0,
            'AttackRating': 0,
            'VelocityPercent': 0,
        }
        self.previous_quests = {'Normal': [], 'Nightmare': [], 'Hell': []}
        self.host = '0.0.0.0'
        self.flask_port = flask_port
        @self.app.route('/', methods=['POST'])
        def receive_data():
            new_data = request.get_json()
            print(new_data)
            self.update_game_state(new_data)
            return jsonify(self.game_state)
        
        @self.app.route('/heartbeat', methods=['GET'])
        def heartbeat():
            # Simply return a successful response indicating the server is alive
            return jsonify({"status": "alive"})

    def update_game_state(self, new_data):
        # Update the values that are present in new_data

        # Update the values that are present in new_data
        for key, value in new_data.items():
            if key == 'KilledMonsters':
                # If the key is 'KilledMonsters', increment the counter instead of updating the value
                self.game_state['KilledMonsters'] += 1
            elif key in self.game_state and isinstance(self.game_state[key], dict) and isinstance(value, dict):
                # Update nested dictionaries
                self.game_state[key].update(value)
            elif key == 'AllQuestsDetails':
                # Calculate the sum of the lengths of the 'CompletedBits' lists for all quests
                self.game_state['QuestPartsCompleted'] = sum(len(quest['CompletedBits']) for quest in value)
            else:
                # Set new value
                self.game_state[key] = value

        # Handle special cases
        if 'Skills' in new_data:
            skills_list = new_data['Skills']
            self.game_state['Skills'] = sum(skill.get('Points', 0) for skill in skills_list)

        if 'Area' in new_data and new_data['Area'] is not None:
            area_added = self.check_new_area(new_data['Area'])
            if area_added:
                self.game_state['NewAreaDiscovered'] = True
            else:
                self.game_state['NewAreaDiscovered'] = False
        
        if 'CompletedQuests' in new_data:
            self.previous_quests = self.game_state['CompletedQuests'].copy()

        #print(f'Game State Contents: {self.game_state}')

    def check_new_area(self, area_id):
        if area_id not in self.game_state['Areas'] and area_id != 1:
            self.game_state['Areas'].append(area_id)
            return True
        return False
    
    def get_state(self):
        return self.game_state.copy()

    def run_server(self):
        # Wrap the Flask app's run method into a Thread
        server_thread = threading.Thread(target=lambda: self.app.run(host=self.host, port=self.flask_port))
        server_thread.start()
        

