class ChooseDifficultyPage extends Page {
  
    CustomSwitchPageButton easyButton;
    CustomSwitchPageButton midButton;
    CustomSwitchPageButton hardButton;
    CustomSwitchPageButton quitButton;
    
    float quitButtonX;
    float quitButtonY;
    float quitButtonWidth;
    float quitButtonHeight;
    
    float hardButtonX;
    float hardButtonY;
    float hardButtonWidth;
    float hardButtonHeight;
    
    float midButtonX;
    float midButtonY;
    float midButtonWidth;
    float midButtonHeight;
    
    float easyButtonX;
    float easyButtonY;
    float easyButtonWidth;
    float easyButtonHeight;

    
    PImage backgroundImage;
    PApplet app;

    ChooseDifficultyPage(PApplet app) {
        this.app = app;
    }

    @Override
    void init() {
        // 設定畫布大小
        windowResize(WIDTH, HEIGHT);

        // 加載背景圖片
        backgroundImage = loadImage("difficultyPage.png");
        backgroundImage.resize(width, height);
        image(backgroundImage, 0, 0);

        // 設定文字樣式
        textSize(round(110 * unit / 3));
        textAlign(CENTER, CENTER);
        fill(255);
        
        // Load music
        player = minim.loadFile("button-hover.wav");
        
        // 按鈕座標與大小變數
        quitButtonX = WIDTH * 269.58 / 1440;
        quitButtonY = HEIGHT * 412.12 / 660;
        quitButtonWidth = WIDTH * 276 / 1440;
        quitButtonHeight = HEIGHT * 276 / 660;
        
        hardButtonX = WIDTH * 509.9 / 1440;
        hardButtonY = HEIGHT * 514.81 / 660;
        hardButtonWidth = WIDTH * 207.4 / 1440;
        hardButtonHeight = HEIGHT * 207.4 / 660;
        
        midButtonX = WIDTH * 699.9 / 1440;
        midButtonY = HEIGHT * 529.81 / 660;
        midButtonWidth = WIDTH * 156 / 1440;
        midButtonHeight = HEIGHT * 156 / 660;
        
        easyButtonX = WIDTH * 849.9 / 1440;
        easyButtonY = HEIGHT * 546.81 / 660;
        easyButtonWidth = WIDTH * 116.4 / 1440;
        easyButtonHeight = HEIGHT * 116.4 / 660;

        
        // 初始化按鈕
        easyButton = new CustomSwitchPageButton(
            easyButtonX, easyButtonY, easyButtonWidth, easyButtonHeight, "",
            (app, aiDifficulty) -> new GamePage(app, 1), app, 1
        );
        easyButton.setButtonDrawer(() -> easyButtonDrawer());
        
        midButton = new CustomSwitchPageButton(
            midButtonX, midButtonY, midButtonWidth, midButtonHeight, "",
            (app, aiDifficulty) -> new GamePage(app, 2), app, 2
        );
        midButton.setButtonDrawer(() -> midButtonDrawer());
        
        hardButton = new CustomSwitchPageButton(
            hardButtonX, hardButtonY, hardButtonWidth, hardButtonHeight, "",
            (app, aiDifficulty) -> new GamePage(app, 3), app, 3
        );
        hardButton.setButtonDrawer(() -> hardButtonDrawer());
        
        quitButton = new CustomSwitchPageButton(
            quitButtonX, quitButtonY, quitButtonWidth, quitButtonHeight, "",
            () -> exit()
        );
        quitButton.setButtonDrawer(() -> quitButtonDrawer());

    }
    
    void quitButtonDrawer() {
        noStroke();
        pushMatrix();
        fill(#e6e6e6);
        translate(quitButtonX, quitButtonY);
        rotate(radians(40));
        rectMode(CENTER);
        rect(0, 0, quitButtonWidth, quitButtonHeight);
        popMatrix();
    
        fill(#000000);
        textSize(29.7 * unit / 3);
        text("QUIT", quitButtonX, quitButtonY);
    }
    
    void hardButtonDrawer() {
        noStroke();
        pushMatrix();
        fill(#ff5b45);
        translate(hardButtonX, hardButtonY);
        rectMode(CENTER);
        rect(0, 0, hardButtonWidth, hardButtonHeight);
    
        fill(#e9e9e9);
        textSize(33.7 * unit / 3);
        text("Hard", 0, 0);
        popMatrix();
    }
    
    void midButtonDrawer() {
        noStroke();
        pushMatrix();
        fill(#ffa82b);
        translate(midButtonX, midButtonY);
        rotate(radians(120));
        rectMode(CENTER);
        rect(0, 0, midButtonWidth, midButtonHeight);
        popMatrix();
    
        fill(#e9e9e9);
        textSize(30.7 * unit / 3);
        text("Mid", midButtonX, midButtonY);
    }
    
    void easyButtonDrawer() {
        noStroke();
        pushMatrix();
        fill(#489900);
        translate(easyButtonX, easyButtonY);
        rotate(radians(160));
        rectMode(CENTER);
        rect(0, 0, easyButtonWidth, easyButtonHeight);
        popMatrix();
    
        fill(#e9e9e9);
        textSize(28.7 * unit / 3);
        text("easy", easyButtonX, easyButtonY);
    }


    @Override
    void render() {
        image(backgroundImage,0,0);
        
        // Render back button
        easyButton.render();
        midButton.render();
        hardButton.render();
        quitButton.render();
    }
      

    @Override
    void onClick() {
        if (easyButton.isMouseOver()) easyButton.onClick();
        if (midButton.isMouseOver()) midButton.onClick();
        if (hardButton.isMouseOver()) hardButton.onClick();
        if (quitButton.isMouseOver()) quitButton.onClick();
    }

    @Override
    void onKeyPressed(int key, int keyCode) {
        // Define behavior if needed
    }
}
