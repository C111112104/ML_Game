class GameOverPage extends Page {

    SwitchPageButton playAgainButton;
    SwitchPageButton backToMenuButton;
    SwitchPageButton quitButton;
    PImage backgroundImage;
    PApplet app;
    int status;
    
    GameOverPage(PApplet app, int status) {
        this.app = app;
        this.status = status;
        if (status == 1) {
            backgroundImage = loadImage("startPageBackground.png"); // if win
        }
    }

    @Override
    void init() {
        // 設定畫布大小
        windowResize(WIDTH, HEIGHT);

        // 加載背景圖片

        backgroundImage = loadImage("gameOverBackground.png"); // default lose
        backgroundImage.resize(width, height);
        image(backgroundImage, 0, 0);

        // 設定文字樣式
        textSize(110 * unit / 3);
        textAlign(CENTER, CENTER);
        fill(255);

        // 在畫布上的指定位置進行數字轉換後繪製 "Tetris"
        text("Tetris", width * 720 / 1440, height * 120 / 660);
        float buttonX = width * (740 - 300 / 2) / 1440;
        float buttonWidth = width * 300 / 1440;
        float buttonHeight = height * 80 / 660;
        
        // Load music
        player = minim.loadFile("button-hover.wav");
        
        // set Button
        playAgainButton = new SwitchPageButton(buttonX, height * (300 - 80 / 2) / 660, buttonWidth, buttonHeight, "Play Again", (app) -> new ChooseDifficultyPage(app), app);
        backToMenuButton = new SwitchPageButton(buttonX, height * (400 - 80 / 2) / 660, buttonWidth, buttonHeight, "Back to Menu", (app) -> new StartPage(app), app);
        quitButton = new SwitchPageButton(buttonX, height * (500 - 80 / 2) / 660, buttonWidth, buttonHeight, "Quit", () -> exit());
        quitButton.initButtonOutlook();
    }

    @Override
    void render() {
        image(backgroundImage,0,0);
        textSize(110 * unit / 3);
        textAlign(CENTER, CENTER);
        fill(255);
        if (status == 1) {
            text("You Win", width * 720 / 1440, height * 120 / 660);
        } else text("You Lose", width * 720 / 1440, height * 120 / 660);
        playAgainButton.render();
        backToMenuButton.render();
        quitButton.render();
    }

    @Override
    void onClick() {
        if (playAgainButton.isMouseOver()) playAgainButton.onClick();
        if (backToMenuButton.isMouseOver()) backToMenuButton.onClick();
        if (quitButton.isMouseOver()) quitButton.onClick();
    }

    @Override
    void onKeyPressed(int key, int keyCode) {
        // Define behavior if needed
    }
}
