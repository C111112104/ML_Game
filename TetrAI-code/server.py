import socket
from ai.dqn_agent import DQNAgent
from ai.tetris import Tetris
import time
import pathlib
import logging

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serversocket.bind(('0.0.0.0', 7000))
serversocket.listen(5) # become a server socket, maximum 5 connections

shapePieceTable = {'I': 0, 'T': 1, 'L': 2, 'J': 3, 'Z': 4, 'S': 5, 'O': 6}
rotationToClient = {0: 0, 90: 1, 180: 2, 270: 3}
agent = DQNAgent(4, modelFile=str(pathlib.Path(__file__).parent.resolve())+"/ai/best.keras")

def getBestAction(board, piece_id):
    env = Tetris(board, piece_id)
    next_states = {tuple(v): k for k, v in env.get_next_states().items()}
    best_state = agent.best_state(next_states.keys())
    best_action = next_states[best_state]
    reward, done = env.play(best_action[0], best_action[1], render=False, render_delay=None)
    # reward = 1
    return best_action[0], best_action[1], reward

def contentToResponse(content: str):
    try:
        shape, boardString = content.split()
        # print(shape)
        board = eval(boardString)
        piece_id = shapePieceTable[shape]
        # print(piece_id, board)
        x, rotation, reward = getBestAction(board, piece_id)
        
        sendString = str(x) + " " +str(rotationToClient[rotation]) + " " + str(reward)
        # print(sendString)
        return sendString.encode()
    except Exception as e:
        ...
        # print(e)
        # print(content)

logging.warning("serverReady")
connection, address = serversocket.accept()
gameState = True
while True:
    try:
        content = connection.recv(1024).decode()
        for c in content.split("END")[:-1]:
            thing = contentToResponse(c)
            try: connection.send(thing)
            except BrokenPipeError: ...
    except ConnectionAbortedError: ...
        