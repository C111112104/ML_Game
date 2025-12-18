import processing.net.*;
import java.util.HashMap;
import java.util.Random;
import java.util.Arrays;
import java.util.Queue;
import java.util.LinkedList;
import java.util.concurrent.Executors;
import java.io.InputStreamReader;
import ddf.minim.*;


void setup() {

    myFont = createFont("AgaveNerdFont-Bold.ttf", round(grid*1.5)); //int(100 * pow((unit / 10), 1.5))
    textFont(myFont);
    frameRate(fps);
    surface.setLocation(displayWidth/2 - WIDTH/2, displayHeight/2 - HEIGHT/2);
    surface.setSize(WIDTH, HEIGHT);
    surface.setTitle("TetrAI");

    // set score table
    score_count = new HashMap<Integer, Integer>();
    score_count.put(0, 0);
    score_count.put(1, 10);
    score_count.put(2, 30);
    score_count.put(3, 50);
    score_count.put(4, 80);
    score_count.put(5,40);
    score_count.put(6,80);
    score_count.put(7,120);
    
    // set piece INFO
    S_Block = loadImage("block-flippedZ.png");
    Z_Block = loadImage("block-Z.png");
    I_Block = loadImage("block-I.png");
    O_Block = loadImage("block-O.png");
    J_Block = loadImage("block-flippedL.png");
    L_Block = loadImage("block-L.png");
    T_Block = loadImage("block-T.png");
    darkFramedBackgoundBlock = loadImage("dark_framed_backgroud.png");
    lightFramedBackgoundBlock = loadImage("dark_framed_backgroud.png");
    interfaceBackgroundBlock = loadImage("interfaceBackgroudBlock.png");
    blockGrayLightBlock = loadImage("scoreBlock.png");
    
    shape_images = new HashMap<String, PImage>();
    shape_images.put("S", S_Block);
    shape_images.put("Z", Z_Block);
    shape_images.put("I", I_Block);
    shape_images.put("O", O_Block);
    shape_images.put("J", J_Block);
    shape_images.put("L", L_Block);
    shape_images.put("T", T_Block);
    shapes = new HashMap<String, ArrayList<int[][]>>();
    shapes.put("S", new ArrayList<int[][]>(Arrays.asList(
        new int[][] { {1, 1}, {2, 1}, {0, 2}, {1, 2} },
        new int[][] { {1, 0}, {1, 1}, {2, 1}, {2, 2} }
    )));
    
    shapes.put("Z", new ArrayList<int[][]>(Arrays.asList(
        new int[][] { {0, 1}, {1, 1}, {1, 2}, {2, 2} },
        new int[][] { {1, 0}, {0, 1}, {1, 1}, {0, 2} }
    )));
    
    shapes.put("I", new ArrayList<int[][]>(Arrays.asList(
        new int[][] { {0, 1}, {1, 1}, {2, 1}, {3, 1} },
        new int[][] { {2, 0}, {2, 1}, {2, 2}, {2, 3} }
    )));
    
    shapes.put("O", new ArrayList<int[][]>(Arrays.asList(
        new int[][] { {0, 1}, {0, 2}, {1, 1}, {1, 2} },
        new int[][] { {0, 1}, {0, 2}, {1, 1}, {1, 2} }
    )));
    
    shapes.put("J", new ArrayList<int[][]>(Arrays.asList(
        new int[][] { {0, 0}, {0, 1}, {1, 1}, {2, 1} },
        new int[][] { {1, 0}, {2, 0}, {1, 1}, {1, 2} },
        new int[][] { {2, 2}, {0, 1}, {1, 1}, {2, 1} },
        new int[][] { {1, 0}, {0, 2}, {1, 1}, {1, 2} }
    )));
    
    shapes.put("L", new ArrayList<int[][]>(Arrays.asList(
        new int[][] { {2, 0}, {0, 1}, {1, 1}, {2, 1} },
        new int[][] { {1, 0}, {1, 1}, {1, 2}, {2, 2} },
        new int[][] { {0, 2}, {0, 1}, {1, 1}, {2, 1} },
        new int[][] { {1, 0}, {1, 1}, {1, 2}, {0, 2} }
    )));
    
    shapes.put("T", new ArrayList<int[][]>(Arrays.asList(
        new int[][] { {1, 0}, {0, 1}, {1, 1}, {2, 1} },
        new int[][] { {1, 0}, {1, 2}, {1, 1}, {2, 1} },
        new int[][] { {1, 2}, {0, 1}, {1, 1}, {2, 1} },
        new int[][] { {1, 0}, {0, 1}, {1, 1}, {1, 2} }
    )));
    
    nextPiecePositions = new ArrayList<int[]>(Arrays.asList(
        new int[] { nextY + 3 * grid, nextX + 1 * grid },
        new int[] { nextY + 8 * grid, nextX + 1 * grid },
        new int[] { nextY + 12 * grid, nextX + 1 * grid }
    ));

    background(backgroundColor);
    
    // initialize env
    minim = new Minim(this);
    pageManager = new PageManager();
    serverhandler = new ServerHandler(this, HOSTNAME, PORT);
    serverhandler.startServer();
    pageManager.switchPage(new StartPage(this));
}

void draw() {
    pageManager.renderCurrentPage();
}

void mousePressed() {
    pageManager.onClickCurrentPage();
}

void keyPressed() {
    pageManager.onKeyPressCurrentPage(key,keyCode);
}
