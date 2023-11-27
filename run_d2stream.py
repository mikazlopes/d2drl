# d2stream.py
from d2stream import D2GameState
## If needed for testing purposes
if __name__ == "__main__":
    game_state_server = D2GameState(flask_port=8123)
    game_state_server.run_server()
