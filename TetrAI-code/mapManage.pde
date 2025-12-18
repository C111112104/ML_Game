class GameBoard {
    /*
    2D-array representing the status of each cell
    0: empty
    1: the moving piece
    2: fixed pieces
    */
    PImage[][] images;
    int[][] status;

    GameBoard() {
        init();
    }

    // Initializes the game board
    void init() {
        images = new PImage[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if ((i + j) % 2 == 0) {
                    images[i][j] = lightFramedBackgoundBlock;
                } else {
                    images[i][j] = darkFramedBackgoundBlock;
                }
            }
        }

        status = new int[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                status[i][j] = 0;
            }
        }
    }

    // Checks if a specific cell is occupied
    boolean isOccupied(int x, int y) {
        if (x < 0 || x >= rows || y < 0 || y >= columns) {
            return true; // Out of bounds counts as occupied
        }
        return status[x][y] == 2; // Fixed pieces are considered occupied
    }
    
    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("[");
        for (int y = 0; y < rows; y++) {
            result.append('[');
            for (int x = 0; x < columns; x++) {
                if (status[y][x] == 2) {
                    result.append("1");
                } else {
                    result.append("0");
                }
                result.append(',');
            }
            result.setLength(result.length() - 1); // Remove last comma
            result.append("],");
        }
        result.setLength(result.length() - 1); // Remove last comma
        result.append(']');
        return result.toString();
    }
}

class LineClearManager {
    GameBoard gameBoard;

    LineClearManager(GameBoard gameBoard) {
        this.gameBoard = gameBoard;
    }

    // Clears full lines and returns the number of lines cleared
    int clearFullLines() {
        int linesCleared = 0;
        for (int i = rows - 1; i >= 0; i--) {
            if (isLineFull(i)) {
                clearLine(i);
                linesCleared++;
                i++; // Recheck the same row since rows above fall down
            }
        }
        return linesCleared;
    }

    // Checks if a specific row is fully occupied
    private boolean isLineFull(int row) {
        for (int j = 0; j < columns; j++) {
            if (gameBoard.status[row][j] != 2) {
                return false;
            }
        }
        return true;
    }

    // Clears a specific line and shifts lines down
    private void clearLine(int row) {
        for (int i = row; i > 0; i--) {
            for (int j = 0; j < columns; j++) {
                gameBoard.status[i][j] = gameBoard.status[i - 1][j];
                gameBoard.images[i][j] = gameBoard.images[i - 1][j];
            }
        }
        // Clear the top row after shifting
        for (int j = 0; j < columns; j++) {
            gameBoard.status[0][j] = 0;
            gameBoard.images[0][j] = (j % 2 == 0) ? lightFramedBackgoundBlock : darkFramedBackgoundBlock;
        }
    }
}

class ScoreManager {
    int lineCount;
    int score;
    boolean prevTspin;
    boolean prevTetris;
    String scoreType;

    ScoreManager() {
        reset();
    }

    // Resets score-related data
    void reset() {
        lineCount = 0;
        score = 0;
        prevTspin = false;
        prevTetris = false;
    }

    // Adds score based on action type and lines cleared
    void addScore(boolean isTspin, int linesCleared) {
        int step = linesCleared + (isTspin ? 5 : 0);
        int basePoint = score_count.getOrDefault(step, 0);
        
        if (!isTspin && linesCleared != 4) scoreType = "normal";

        if (isTspin) {
            if (prevTspin) {
                basePoint *= 1.5;
                scoreType = "B2B Tspin";
            } else scoreType = "Tspin";
        } else if (linesCleared == 4) {
            if (prevTetris) {
                basePoint *= 1.5;
                scoreType = "B2B Tetris";
            } else scoreType = "Tetris";
        }
        
        if (0 < linesCleared && linesCleared < 4) {
            player = minim.loadFile("tetris-single-clear.wav");
            player.play(0);
            prevTspin = isTspin;
            prevTetris = linesCleared == 4;
        }
        
        if (isTspin) {
            player = minim.loadFile("tetris-sound.wav");
            player.play(0);
            prevTspin = true;
        }
        
        if (linesCleared == 4) {
            player = minim.loadFile("tetris-sound.wav");
            player.play(0);
            prevTetris = true;
        }

        score += basePoint;
        lineCount += linesCleared;

    }
}
