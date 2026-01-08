// Constants
final float unit = 5; // 基本單位 you can make this value larger if game interface is too small
final int grid = int(6 * unit);
final int rows = 20;
final int columns = 10;
final color backgroundColor = color(33, 47, 60);
final int WIDTH = (7 * grid + columns * grid + 7 * grid) * 2;
final int HEIGHT = 1 * grid + rows * grid + 1 * grid;
final int gameBoardX = grid*7;
final int gameBoardY = grid*1;
final int nextX = grid * 18;
final int nextY = grid * 2;
final int holdY = grid * 2;
final int holdX = grid * 1;
final int scoreX = grid * 1;
final int scoreY = grid * 16;
final boolean DEBUG = false;
final String HOSTNAME = "0.0.0.0";
final int PORT = 7000;

// global variable
int fps = 60;
int difficulty = 20; //調整降下的速度，數字越大會越慢
HashMap<String, ArrayList<int[][]>> shapes;
HashMap<String, PImage> shape_images;
HashMap<Integer, Integer> score_count;

PFont myFont;
int fontSize = (int)(100 * pow((unit / 10.0), 1.5));
PVector score_pos = new PVector(columns * grid + 10 * unit, WIDTH / 2);
PVector line_pos = new PVector(columns * grid + 10 * unit, HEIGHT / 2 + fontSize);
PVector next_piece_pos = new PVector(columns * grid + 22 * unit, HEIGHT / 2 - 30 * unit);
ArrayList<int[]> nextPiecePositions;

Minim minim;
AudioPlayer player;
PageManager pageManager;
ServerHandler serverhandler;

PImage S_Block;
PImage Z_Block;
PImage I_Block;
PImage O_Block;
PImage J_Block;
PImage L_Block;
PImage T_Block;
PImage darkFramedBackgoundBlock;
PImage lightFramedBackgoundBlock;
PImage interfaceBackgroundBlock;
PImage blockGrayLightBlock;
