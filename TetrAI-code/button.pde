import java.util.function.Supplier;
import java.util.function.Function;
import java.util.function.BiFunction;

abstract class Button {
    float x, y, width, height;
    PImage backgroundImage;
    String text;
    int textSize;
    int textAlignX, textAlignY;
    int textColor;
    int backgroundColor;
    boolean onButton;
    Runnable drawer;

    // Constructor with customizable properties
    Button(float x, float y, float width, float height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.backgroundImage = null;
        this.text = "";
        this.textSize = round(16 * unit / 3);
        this.textAlignX = CENTER;
        this.textAlignY = CENTER;
        this.textColor = 0;
        this.backgroundColor = 200;
        this.onButton = false;
        this.drawer = () -> defaultButtonDrawer();
    }

    // Setters for optional properties
    void setBackgroundImage(PImage backgroundImage) {
        this.backgroundImage = backgroundImage;
    }

    void setText(String text, int textSize, int textAlignX, int textAlignY) {
        this.text = text;
        this.textSize = textSize;
        this.textAlignX = textAlignX;
        this.textAlignY = textAlignY;
    }

    void setBackgroundColor(int backgroundColor) {
        this.backgroundColor = backgroundColor;
    }
    
    void setButtonDrawer(Runnable drawer) {
        this.drawer = drawer;
    }

    // Checks if the mouse is over the button
    boolean isMouseOver() {
        return mouseX >= x && mouseX <= x + width && mouseY >= y && mouseY <= y + height;
    }
    
    void defaultButtonDrawer() {
        fill(backgroundColor);
        rect(x,y,width,height);
    }
    
    // Renders the button
    void render() {
        pushStyle();
        if (isMouseOver() && !onButton) {
            onTop();
            onButton = true;
        } else if (!isMouseOver() && onButton) {
            onLeaveButton();
            onButton = false;
        }

        // Draw background
        if (backgroundImage != null) {
            image(backgroundImage, x, y, width, height);
        } else {
            drawer.run();
        }

        // Draw text
        if (text != null && !text.isEmpty()) {
            textAlign(textAlignX, textAlignY);
            textSize(textSize);
            fill(textColor);
            text(text, x + width / 2, y + height / 2);
        }

        popStyle();
    }

    // Abstract methods for events
    abstract void onTop();
    
    abstract void onLeaveButton();

    abstract void onClick();
}

class SwitchPageButton extends Button {
    private Supplier<Page> noParamSupplier;
    private Function<PApplet, Page> singleParamSupplier;
    private BiFunction<PApplet, Integer, Page> doubleParamSupplier;
    private Runnable actionSupplier;

    private PApplet app;
    private int param1;

    private PImage buttonImage;
    private PImage hoveredButtonImage;

    // 無參數建構器
    SwitchPageButton(float x, float y, float width, float height, String text, Supplier<Page> noParamSupplier) {
        super(x, y, width, height);
        this.text = text;
        this.noParamSupplier = noParamSupplier;
        initButtonOutlook();
    }

    // 單參數建構器
    SwitchPageButton(float x, float y, float width, float height, String text, Function<PApplet, Page> singleParamSupplier, PApplet app) {
        super(x, y, width, height);
        this.text = text;
        this.singleParamSupplier = singleParamSupplier;
        this.app = app;
        initButtonOutlook();
    }

    // 雙參數建構器
    SwitchPageButton(float x, float y, float width, float height, String text, BiFunction<PApplet, Integer, Page> doubleParamSupplier, PApplet app, int param1) {
        super(x, y, width, height);
        this.text = text;
        this.doubleParamSupplier = doubleParamSupplier;
        this.app = app;
        this.param1 = param1;
    }
    
    // Runnable 參數建構器
    SwitchPageButton(float x, float y, float width, float height, String text, Runnable actionSupplier) {
        super(x, y, width, height);
        this.text = text;
        this.actionSupplier = actionSupplier;
    }

    // 初始化按鈕樣式
    void initButtonOutlook() {
        this.textSize = round(30 * unit / 3);
        this.textColor = 255;
        this.buttonImage = loadImage("startPageButton.png");
        this.hoveredButtonImage = loadImage("startPageButtonHovered.png");
        setBackgroundImage(buttonImage);
    }

    @Override
    void onTop() {
        setBackgroundImage(hoveredButtonImage);
        player.play(0);
    }

    @Override
    void onLeaveButton() {
        setBackgroundImage(buttonImage);
    }

    @Override
    void onClick() {
        player = minim.loadFile("button-click.wav");
        player.play();

        if (noParamSupplier != null) {
            pageManager.switchPage(noParamSupplier.get());  // 無參數頁面
        } else if (singleParamSupplier != null) {
            pageManager.switchPage(singleParamSupplier.apply(app));  // 單參數頁面
        } else if (doubleParamSupplier != null) {
            pageManager.switchPage(doubleParamSupplier.apply(app, param1));  // 雙參數頁面
        } else if(actionSupplier != null) {
            actionSupplier.run();
        }
    }
}

class CustomSwitchPageButton extends SwitchPageButton {

    Runnable actionSupplier;

    // Runnable 參數建構器
    CustomSwitchPageButton(float centerX, float centerY, float width, float height, String text, Runnable actionSupplier) {
        super(centerX, centerY, width, height, text, actionSupplier);
    }

    // 無參數建構器
    CustomSwitchPageButton(float centerX, float centerY, float width, float height, String text, Supplier<Page> noParamSupplier) {
        super(centerX, centerY, width, height, text, noParamSupplier);
    }

    // 單參數建構器
    CustomSwitchPageButton(float centerX, float centerY, float width, float height, String text, Function<PApplet, Page> singleParamSupplier, PApplet app) {
        super(centerX, centerY, width, height, text, singleParamSupplier, app);
    }

    // 雙參數建構器
    CustomSwitchPageButton(float centerX, float centerY, float width, float height, String text, BiFunction<PApplet, Integer, Page> doubleParamSupplier, PApplet app, int param1) {
        super(centerX, centerY, width, height, text, doubleParamSupplier, app, param1);
    }

    // 重寫判斷滑鼠是否在按鈕上
    @Override
    boolean isMouseOver() {
        float halfWidth = width / 2;
        return dist(mouseX, mouseY, x, y) < halfWidth;
    }
    
    // notChange image onTop
    @Override
    void onTop() {
        player.play(0);
    }

    @Override
    void render() {
        pushStyle();
        if (isMouseOver() && !onButton) {
            onTop();
            onButton = true;
        } else if (!isMouseOver() && onButton) {
            onLeaveButton();
            onButton = false;
        }
        
        if (drawer != null) drawer.run();  // 使用自定義繪製器
        
        if (text != null && !text.isEmpty()) {
            textAlign(textAlignX, textAlignY);
            textSize(textSize);
            fill(textColor);
            text(text, x, y);
        }
        popStyle();
    }
}
