class AboutPage extends Page {
  
    SwitchPageButton backButton;
    PImage backgroundImage;
    PApplet app;

    AboutPage(PApplet app) {
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
        float buttonY = height * (560 - 80 / 2) / 660;  // 原來是 660，改為 560 上調 100
        
        // setButton
        backButton = new SwitchPageButton(buttonX, buttonY, buttonWidth, buttonHeight, "BACK", (app) -> new StartPage(app), app);
    }

    @Override
    void render() {
        image(backgroundImage,0,0);
        
        fill(255); // 設定文字顏色為白色
        textSize(35 * unit / 3); // 設定文字大小為35
        textAlign(CENTER, CENTER); // 設定文字對齊方式為居中
        text("About us", width * 720 / 1440, height * 60 / 660);  
       
        textSize(20 * unit / 3);
        text("This game is made with love and passion.", width * 720 / 1440, height * 110 / 660);
        
        textSize(25 * unit / 3);
        text("[Acknowledgement]", width * 720 / 1440, height * 180 / 660);
        
        textSize(25 * unit / 3);
        text("[Team]", width * 300 / 1440, height * 180 / 660);
        
        textSize(25 * unit / 3);
        text("- Copyright Notice -", width * 1140 / 1440, height * 180 / 660);
        
        textSize(23 * unit / 3);
        text("Copyright © ", width * 1140 / 1440, height * 240 / 660);
        
        textSize(23 * unit / 3);
        text("All Rights Reserved.", width * 1140 / 1440, height * 270 / 660);
        
        textSize(18 * unit / 3);
        text("All images used in this ", width * 1140 / 1440, height * 340 / 660);
        
        textSize(18 * unit / 3);
        text("work are original, created ", width * 1140 / 1440, height * 360 / 660);
        
        text("by members of the team.", width * 1140 / 1440, height * 380 / 660);
        
        text("This work is not intended", width * 1140 / 1440, height * 410 / 660);
        
        text("for any commercial use and", width * 1140 / 1440, height * 430 / 660);
        
        text("is solely provided for teaching", width * 1140 / 1440, height * 450 / 660);
        
        text("and learning purposes.", width * 1140 / 1440, height * 470 / 660);
        
        textSize(19 * unit / 3);
        text("Ren-Chun Yao - Team Leader -", width * 300 / 1440, height * 220 / 660);
        
        textSize(15 * unit / 3);
        text(" - Main Programmer - ", width * 300 / 1440, height * 240 / 660);
        
        textSize(15 * unit / 3);
        text(" - Schedule Coordination - ", width * 300 / 1440, height * 260 / 660);
        
        textSize(18 * unit / 3);
        text("Tzu-En Peng - Team Member -", width * 300 / 1440, height * 290 / 660);
        
        textSize(15 * unit / 3);
        text(" - Lead Art Designer - ", width * 300 / 1440, height * 310 / 660);
        
         textSize(15 * unit / 3);
        text(" - Programmer - ", width * 300 / 1440, height * 330 / 660);
        
        textSize(17 * unit / 3);
        text("Cheng-Yin Lu - Team Member -", width * 300 / 1440, height * 360 / 660);
        
        textSize(15 * unit / 3);
        text(" - Interface Design Support - ", width * 300 / 1440, height * 380 / 660);
        
        textSize(15 * unit / 3);
        text(" - Art Design Support - ", width * 300 / 1440, height * 400 / 660);
        
        textSize(17 * unit / 3);
        text("Meng-Hsun Tsai - Team Member -", width * 300 / 1440, height * 430 / 660);
        
        textSize(15 * unit / 3);
        text(" - Programming Support - ", width * 300 / 1440, height * 450 / 660);
        
        textSize(15 * unit / 3);
        text(" - Code Testing - ", width * 300 / 1440, height * 470 / 660);
        
        textSize(17 * unit / 3);
        text("Ting-Wei Lin - Team Member -", width * 300 / 1440, height * 500 / 660);
        
        textSize(15 * unit / 3);
        text(" - Program Testing Support - ", width * 300 / 1440, height * 520 / 660);
        
        textSize(18 * unit / 3);
        text("Thanks to -Nuno Faria-", width * 720 / 1440, height * 220 / 660);
        
        textSize(18 * unit / 3);
        text("for the AI training code", width * 720 / 1440, height * 260 / 660);
        
        textSize(18 * unit / 3);
        text("and -Alexey Pajitnov-", width * 720 / 1440, height * 300 / 660);
        
        textSize(18 * unit / 3);
        text("for inventing Tetris.", width * 720 / 1440, height * 340 / 660);
    
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
