class GamePage extends Page {
    Agent agent;
    Player player;
    int counter;
    int difficulty;
    long currMilisec;
    RenderTetris renderPlayer;
    RenderTetris renderAgent;
    AnimationManager animationManager;
    PApplet app;

    // Constructor initializing the game page
    GamePage(PApplet app, int aiDifficulty) {
        this.agent = new Agent(serverhandler, aiDifficulty);
        this.player = new Player();
        this.counter = 0;
        this.difficulty = (4 - aiDifficulty) * 10;
        this.currMilisec = millis();
        this.renderPlayer = new RenderTetris(player.tetrisGame, 0, 0);
        this.renderAgent = new RenderTetris(agent.tetrisGame, grid * 24, 0);
        this.animationManager = new AnimationManager();
        this.app = app;
    }

    // Initializes the game page
    @Override
    void init() {
        windowResize(WIDTH, HEIGHT);
        
    }

    // Renders the game page
    @Override
    void render() {
        if (Math.abs(player.tetrisGame.scoreManager.score - agent.tetrisGame.scoreManager.score) >= 300) {
            if (player.tetrisGame.scoreManager.score > agent.tetrisGame.scoreManager.score) {
                gameEnd("Win");
            } else gameEnd("Lose");
        }

        if (counter == difficulty && !DEBUG) {
            player.tetrisGame.moveHandler.drop();
            agent.tetrisGame.moveHandler.drop();
            counter -= difficulty;
        }

        if (player.tetrisGame.pieceManager.current.is_fixed) {
            player.onFixed();
            if (!player.tetrisGame.scoreManager.scoreType.equals("normal"))
            showAnimation("player",player.tetrisGame.scoreManager.scoreType,millis());
        }

        if (agent.tetrisGame.pieceManager.current.is_fixed) {
            agent.onFixed();
            if (!agent.tetrisGame.scoreManager.scoreType.equals("normal"))
            showAnimation("agent",agent.tetrisGame.scoreManager.scoreType,millis());
        }

        if (player.tetrisGame.isDefeat()) {
            gameEnd("Lose");
        }

        if (agent.tetrisGame.isDefeat()) {
            gameEnd("Win");
        }
        
        renderGame();
        animationManager.renderAnimations();
        if (!player.tetrisGame.gameStart) {
            startCountDown();
        }

        if (player.tetrisGame.gameStart) {
            agent.doAILogic();
            counter++;
        }
        
        // Decrement key ticker values
        player.keyTicker.replaceAll((k, v) -> Math.max(v - 1, 0));
    }

    // Handles click events
    @Override
    void onClick() {
        // Define click behavior if needed
    }
    
    // Handles click events
    @Override
    void onKeyPressed(int key, int keyCode) {
        player.onKeyPressed(key, keyCode);
    }
    
    // Game end logic
    void gameEnd(String result) {
        int isWin = result.equals("Win") ? 1 : 0;
        serverhandler.client.clear();
        player.tetrisGame.gameStart = false;
        pageManager.switchPage(new GameOverPage(app, isWin));
    }
    
    // render player and agent tetris board
    void renderGame() {
        renderPlayer.render("you");
        renderAgent.render("ai");
    }

    // Displays animation
    void showAnimation(String animationDisplayOn, String animationContent, int currMillis) {
        int width = animationDisplayOn.equals("player") ? WIDTH / 4 : WIDTH * 3 / 4;
        int height = HEIGHT / 2;
    
        TextAnimation newAnimation = new TextAnimation(currMillis, 1000, animationContent, width, height);
        animationManager.addAnimation(newAnimation);
    }

    // Starts the countdown
    void startCountDown() {
        if (millis() - currMilisec >= 3000) {
            player.tetrisGame.gameStart = true;
            agent.myClient.write(agent.tetrisGame.pieceManager.current.shape + " " + agent.tetrisGame.gameBoard.toString() + "END");
            if (agent.aiDifficulty == 3) {
                agent.myClient.write(agent.tetrisGame.pieceManager.nextQueue.peek().shape + " " + agent.tetrisGame.gameBoard.toString() + "END");
            }
        } else {
            pushMatrix();
            textAlign(CENTER);
            fill(255, 255, 0);
            int countdown = 3 - (int) ((millis() - currMilisec) / 1000);
            text(nf(countdown, 1), WIDTH / 4, HEIGHT / 2, 48); // 顯示在玩家
            text(nf(countdown, 1), WIDTH * 3 / 4, HEIGHT / 2, 48); // 顯示在AI
            popMatrix();
        }
    }
    
}
