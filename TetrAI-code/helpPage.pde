class HelpPage extends Page {
  
    SwitchPageButton backButton;
    PImage backgroundImage;
    PApplet app;

    HelpPage(PApplet app) {
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
        
        // Load music
        player = minim.loadFile("button-hover.wav");
        
        // BackButton coorodinate
        float buttonX = width * (740 - 300 / 2) / 1440;
        float buttonWidth = width * 300 / 1440;
        float buttonHeight = height * 80 / 660;  
        float buttonY = height * (460 - 80 / 2) / 660;
        
        // setButton
        backButton = new SwitchPageButton(buttonX, buttonY, buttonWidth, buttonHeight, "BACK", (app) -> new StartPage(app), app);
    }

    @Override
    void render() {
        image(backgroundImage,0,0);
        // Set text styles
        fill(255); // Set text color to white
        textSize(35 * unit / 3); // Set text size to 35
        textAlign(CENTER, CENTER); // Center text alignment
        text("Help", width * 720 / 1440, height * 40 / 660); // Moved up by 20
        textSize(20 * unit / 3);
        text("This game is made with love and passion.", width * 720 / 1440, height * 90 / 660); // Moved up by 20
        textSize(25 * unit / 3);
        text("Win/Lose Conditions", width * 720 / 1440, height * 160 / 660); // Moved up by 20
        textSize(23 * unit / 3);
        text("Scoring Breakdown", width * 300 / 1440, height * 160 / 660); // Moved up by 20
        textSize(23 * unit / 3);
        text("- Control -", width * 1140 / 1440, height * 160 / 660); // Moved up by 20
        
        textSize(20 * unit / 3);
        text("Up Arrow: Rotate clockwise", width * 1140 / 1440, height * 230 / 660); // Moved up by 20
        textSize(18 * unit / 3);
        text("Down Arrow: Move down 1 space", width * 1140 / 1440, height * 280 / 660); // Moved up by 20
        text("Left Arrow: Move left 1 space", width * 1140 / 1440, height * 330 / 660); // Moved up by 20
        text("Right Arrow: Move right 1 space", width * 1140 / 1440, height * 380 / 660); // Moved up by 20
        text("Z Key: Rotate counterclockwise", width * 1140 / 1440, height * 430 / 660); // Moved up by 20
        text("Shift (Left/Right): Hold piece", width * 1140 / 1440, height * 480 / 660); // Moved up by 20
        text(" (swap if already holding)", width * 1140 / 1440, height * 510 / 660); // Moved up by 20
        textSize(20 * unit / 3);
        text("Points earned for     ", width * 300 / 1440, height * 200 / 660); // Moved up by 20
        textSize(20 * unit / 3);
        text("various actions:", width * 300 / 1440, height * 230 / 660); // Moved up by 20
        
        textSize(21 * unit / 3);
        text("1.Line Clears", width * 300 / 1440, height * 280 / 660); // Moved up by 20
        textSize(15 * unit / 3);
        text("1  line: 10 points", width * 300 / 1440, height * 320 / 660); // Moved up by 20
        textSize(15 * unit / 3);
        text("2 lines: 30 points", width * 300 / 1440, height * 340 / 660); // Moved up by 20
        textSize(15 * unit / 3);
        text("3 lines: 50 points", width * 300 / 1440, height * 360 / 660); // Moved up by 20
        textSize(15 * unit / 3);
        text("4 lines (Tetris): 80 points", width * 300 / 1440, height * 380 / 660); // Moved up by 20
        textSize(21 * unit / 3);
        text("2.T-Spin", width * 300 / 1440, height * 430 / 660); // Moved up by 20
        textSize(15 * unit / 3);
        text("0 lines: 40 points", width * 300 / 1440, height * 460 / 660); // Moved up by 20
        textSize(15 * unit / 3);
        text("1 line: 80 points", width * 300 / 1440, height * 480 / 660); // Moved up by 20
        text("2 lines: 120 points", width * 300 / 1440, height * 500 / 660); // Moved up by 20
        textSize(21 * unit / 3);
        text("3.Back-to-Back Bonus", width * 300 / 1440, height * 550 / 660); // Moved up by 20
        textSize(15 * unit / 3);
        text("B2B T-Spin: 1.5x current T-Spin score", width * 300 / 1440, height * 580 / 660); // Moved up by 20
        text("B2B Tetris: 1.5x current Tetris score", width * 300 / 1440, height * 600 / 660); // Moved up by 20
        textSize(18 * unit / 3);
        text("Lose: Blocks reach the top  ", width * 720 / 1440, height * 220 / 660); // Moved up by 20
        textSize(18 * unit / 3);
        text("Win: Score more than AI  ", width * 720 / 1440, height * 270 / 660); // Moved up by 20
        textSize(18 * unit / 3);
        text("(e.g., 300 points).", width * 720 / 1440, height * 295 / 660); // Moved up by 20
        textSize(18 * unit / 3);
        text("Lose if AI scores more.", width * 720 / 1440, height * 320 / 660); // Moved up by 20
        
        // Render back button
        backButton.render();
    }
      

    @Override
    void onClick() {
        if (backButton.isMouseOver()) backButton.onClick();
    }

    @Override
    void onKeyPressed(int key, int keyCode) {
        // Define behavior if needed
    }
}
