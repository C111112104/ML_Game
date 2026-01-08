abstract class Animation {

    private final int startTime;
    private final int duration;

    // Constructor to initialize start time
    public Animation(int startTime, int duration) {
        this.startTime = startTime;
        this.duration = duration;
    }
        

    // Abstract method for rendering the animation
    public int getStartTime() {
        return startTime;
    }

    public int getDuration() {
        return duration;
    }

    abstract void render();
}

class TextAnimation extends Animation {
    private final String animationContent;
    private final int textColor;
    private final int textAlignX;
    private final int textAlignY;
    private final int width;
    private final int textSize;
    private final int height;

    public TextAnimation(int startTime, int duration, String animationContent, int width, int height) {
        this(startTime, duration, animationContent, width, height, color(255, 255, 0), CENTER, CENTER, 32);
    }

    public TextAnimation(int startTime, int duration, String animationContent, int width, int height, int textColor, int textAlignX, int textAlignY, int textSize) {
        super(startTime, duration);
        
        this.animationContent = animationContent;
        this.width = width;
        this.height = height;
        this.textColor = textColor;
        this.textAlignX = textAlignX;
        this.textAlignY = textAlignY;
        this.textSize = textSize;
    }

    @Override
    void render() {
        pushStyle();
        textAlign(textAlignX, textAlignY);
        fill(textColor);
        textSize(textSize);
        text(animationContent, width, height);
        popStyle();
    }
}

class AnimationManager {
    private final Queue<Animation> animations;

    public AnimationManager() {
        animations = new LinkedList<>();
    }

    public void addAnimation(Animation animation) {
        animations.offer(animation);
    }

    public void renderAnimations() {
        int currentMillis = millis();
        while (!animations.isEmpty() && currentMillis - animations.peek().getStartTime() > animations.peek().getDuration()) {
            animations.poll();
        }
        for (Animation animation : animations) {
            animation.render();
        }
    }
}
