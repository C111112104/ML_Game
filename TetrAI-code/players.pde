class Player {
    TetrisGame tetrisGame;
    HashMap<Integer, Integer> keyTicker;

    // Constructor that initializes the player with a TetrisGame instance
    Player() {
        this.tetrisGame = new TetrisGame();
        this.keyTicker = new HashMap<Integer, Integer>();
    }

    // Handles onFix
    void onFixed() {
        tetrisGame.onFixed();
    }

    // Handles key press events
    void onKeyPressed(int key, int keyCode) {
        if (!tetrisGame.gameStart) return;

        switch (keyCode) {
            case UP:
                tetrisGame.moveHandler.rotate(false);
                break;
            case 90: // z button
                tetrisGame.moveHandler.rotate(true);
                break;
            case DOWN:
                keyTicker.put(DOWN, 13);
                tetrisGame.moveHandler.drop();
                break;
            case LEFT:
                keyTicker.put(LEFT, 13);
                tetrisGame.moveHandler.moveLeft();
                break;
            case RIGHT:
                keyTicker.put(RIGHT, 13);
                tetrisGame.moveHandler.moveRight();
                break;
            case SHIFT:
                keyTicker.put(SHIFT, 13);
                tetrisGame.pieceManager.hold();
                break;
        }

        if (key == ' ') {
            tetrisGame.moveHandler.instantDrop();
        }
    }
}


class Agent {
    int aiDifficulty;
    boolean isMoveDone;
    int pieceAgentX;
    int pieceAgentRotation;
    Queue<int[]> actions = new LinkedList<int[]>();
    ServerHandler myClient;
    TetrisGame tetrisGame;

    // Constructor initializing the agent
    Agent(ServerHandler serverHandler, int aiDifficulty) {
        this.tetrisGame = new TetrisGame();
        this.aiDifficulty = aiDifficulty;
        this.isMoveDone = true;
        this.myClient = serverHandler;
        while (serverhandler.available() > 0)
            serverhandler.readString();
    }

    // AI Logic Execution
    void doAILogic() {
        if (myClient.available() > 0) {
            String response = myClient.readString();
            String[] action = response.split(" ");
            pieceAgentX = Integer.parseInt(action[0]);
            pieceAgentRotation = Integer.parseInt(action[1]);
            int reward = Integer.parseInt(action[2]);
            
            if (tetrisGame.pieceManager.current.shape.equals("I")) {
                tetrisGame.moveHandler.drop();
                tetrisGame.moveHandler.drop();
            }
            
            switch (aiDifficulty) {
                
                case 1:
                    tetrisGame.pieceManager.current.x = pieceAgentX;
                    tetrisGame.pieceManager.current.rotation = pieceAgentRotation;
                    break;
                case 2:
                    tetrisGame.pieceManager.current.rotation = pieceAgentRotation;
                    break;
                case 3:
                    actions.offer(new int[]{pieceAgentX, pieceAgentRotation, reward});
                    break;
            }
            isMoveDone = false;
        }

        if (isMoveDone) return;

        switch (aiDifficulty) {
            case 1:
                if (frameCount % 10 == 0) {
                    tetrisGame.moveHandler.drop();
                }
                break;
            case 2:
                if (frameCount % 10 == 0) {
                    if (tetrisGame.pieceManager.current.x < pieceAgentX) {
                        tetrisGame.moveHandler.moveRight();
                    } else if (tetrisGame.pieceManager.current.x > pieceAgentX) {
                        tetrisGame.moveHandler.moveLeft();
                    } else {
                        tetrisGame.moveHandler.instantDrop();
                        isMoveDone = true;
                    }
                }
                break;
            case 3:
                if (actions.size() >= 2) {
                    int[] currMove = actions.poll();
                    int[] holdMove = actions.poll();

                    if (currMove[2] < holdMove[2]) {
                        tetrisGame.pieceManager.hold();
                        tetrisGame.pieceManager.current.rotation = holdMove[1];
                        tetrisGame.pieceManager.current.x = holdMove[0];
                    } else {
                        tetrisGame.pieceManager.current.rotation = currMove[1];
                        tetrisGame.pieceManager.current.x = currMove[0];
                    }
                    tetrisGame.moveHandler.instantDrop();
                    isMoveDone = true;
                }
                break;
        }
    }

    // Handles piece fixation and game updates
    void onFixed() {
        tetrisGame.onFixed();
        myClient.write(tetrisGame.pieceManager.current.shape + " " + tetrisGame.gameBoard.toString() + "END");

        if (aiDifficulty == 3) {
            if (tetrisGame.pieceManager.hold == null) {
                myClient.write(tetrisGame.pieceManager.nextQueue.peek().shape + " " + tetrisGame.gameBoard.toString() + "END");
            } else if (!tetrisGame.pieceManager.holdLocked) {
                myClient.write(tetrisGame.pieceManager.hold.shape + " " + tetrisGame.gameBoard.toString() + "END");
            } else {
                myClient.write(tetrisGame.pieceManager.current.shape + " " + tetrisGame.gameBoard.toString() + "END");
            }
        }
    }
}
