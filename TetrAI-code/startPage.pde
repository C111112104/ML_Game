class StartPage extends Page {

    SwitchPageButton enterButton;
    SwitchPageButton helpButton;
    SwitchPageButton aboutUsButton;
    PImage backgroundImage;
    PApplet app;
    
    StartPage(PApplet app) {
        this.app = app;
    }

    @Override
    void init() {
        // 設定畫布大小
        windowResize(WIDTH, HEIGHT);

        // 加載背景圖片
        backgroundImage = loadImage("startPageBackground.png");
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
        
        enterButton = new SwitchPageButton(buttonX, height * (300 - 80 / 2) / 660, buttonWidth, buttonHeight, "ENTER", (app) -> new ChooseDifficultyPage(app), app);
        helpButton = new SwitchPageButton(buttonX, height * (400 - 80 / 2) / 660, buttonWidth, buttonHeight, "HELP", (app) -> new HelpPage(app), app);
        aboutUsButton = new SwitchPageButton(buttonX, height * (500 - 80 / 2) / 660, buttonWidth, buttonHeight, "ABOUT US", (app) -> new AboutPage(app), app);
    }

    @Override
    void render() {
        image(backgroundImage,0,0);
        textSize(110 * unit / 3);
        textAlign(CENTER, CENTER);
        fill(255);
        text("TetrAI", width * 720 / 1440, height * 120 / 660);

        enterButton.render();
        helpButton.render();
        aboutUsButton.render();
    }

    @Override
    void onClick() {
        if (enterButton.isMouseOver()) enterButton.onClick();
        if (helpButton.isMouseOver()) helpButton.onClick();
        if (aboutUsButton.isMouseOver()) aboutUsButton.onClick();
    }

    @Override
    void onKeyPressed(int key, int keyCode) {
        // Define behavior if needed
    }
}
