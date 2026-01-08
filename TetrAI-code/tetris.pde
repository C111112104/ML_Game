class TetrisGame {
    GameBoard gameBoard;
    LineClearManager lineClearManager;
    ScoreManager scoreManager;
    PieceManager pieceManager;
    MoveHandler moveHandler;
    boolean gameStart;

    TetrisGame() {
        reset();
    }
    
    void reset() {
        this.gameBoard = new GameBoard();
        this.lineClearManager = new LineClearManager(gameBoard);
        this.scoreManager = new ScoreManager();
        this.pieceManager = new PieceManager();
        this.moveHandler = new MoveHandler(gameBoard, pieceManager);
        this.gameStart = false; // Default game state is stopped
    }

    // Determines if the current piece is performing a T-Spin
    boolean isTspin() {
        if (!pieceManager.current.shape.equals("T")) return false;  // Must be a T piece

        ArrayList<int[]> positions = pieceManager.current.getCellsAbsolutePosition();
        int x = positions.get(2)[1];  // Center X coordinate
        int y = positions.get(2)[0];  // Center Y coordinate

        if (y + 1 >= gameBoard.status.length || y < 1) return false;  // Bottom or top out of bounds
        if (x - 1 < 0 || x + 1 >= gameBoard.status[0].length) return false;  // Left or right out of bounds

        int occupiedCorners = 0;
        if (gameBoard.isOccupied(y - 1, x - 1)) occupiedCorners++;  // Top-left
        if (gameBoard.isOccupied(y - 1, x + 1)) occupiedCorners++;  // Top-right
        if (gameBoard.isOccupied(y + 1, x - 1)) occupiedCorners++;  // Bottom-left
        if (gameBoard.isOccupied(y + 1, x + 1)) occupiedCorners++;  // Bottom-right

        boolean caseSpecialFactor = true;
        if (abs(pieceManager.current.rotation % 4) == 0 && !gameBoard.isOccupied(y - 1, x - 1)) caseSpecialFactor = false; // Top-left and 0 rotation
        if (abs(pieceManager.current.rotation % 4) == 2 && !gameBoard.isOccupied(y - 1, x + 1)) caseSpecialFactor = false; // Top-right and 2 rotation

        return occupiedCorners >= 3 && caseSpecialFactor;
    }

    // Determines if the player is defeated
    boolean isDefeat() {
        for (int[] pos : pieceManager.current.getCellsAbsolutePosition()) {
            int x = pos[1];
            int y = pos[0];
            if (gameBoard.status[1][5] == 2) {
                return true;
            }
        }
        return false;
    }
    
    void onFixed() {
        boolean isTspin = isTspin();
        int linesCleared = lineClearManager.clearFullLines();
        scoreManager.addScore(isTspin, linesCleared);
        pieceManager.onFixed();
    }
}

class RenderTetris {

    TetrisGame tetrisGame;
    int positionX;
    int positionY;

    RenderTetris(TetrisGame tetrisGame, int positionX, int positionY) {
        this.tetrisGame = tetrisGame;
        this.positionX = positionX;
        this.positionY = positionY;
    }
    
    // Draws interface background
    void drawFullBackGround() {
        for (int y = 0; y < 22; y++) {
              for (int x = 0; x < 24; x++) {
                image(interfaceBackgroundBlock, positionX + x * grid, positionY + y * grid, grid, grid);
              }
        }
    }

    // Draws function board background
    void drawFunctionBoardBackground(int y, int x, int positionY, int positionX) {
        if ((y+x)%2==0) image(darkFramedBackgoundBlock, positionX + x * grid, positionY + y * grid, grid, grid);
        else image(lightFramedBackgoundBlock, positionX + x * grid, positionY + y * grid, grid, grid);
    }

    // Draws game board background
    void drawGameBoardBackground(int y, int x, int gameBoardY, int gameBoardX) {
        if ((x + y) % 2 == 0) {
            image(lightFramedBackgoundBlock, gameBoardX + x * grid, gameBoardY + y * grid, grid, grid);
        } else {
            image(darkFramedBackgoundBlock, gameBoardX + x * grid, gameBoardY + y * grid, grid, grid);
        }
    }

    void clearPreviousPiece() {
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < columns; x++) {
                if (tetrisGame.gameBoard.status[y][x] != 2) {
                    tetrisGame.gameBoard.status[y][x] = 0;
                }
            }
        }
    }

    void displayCurrentPiece() {
        Piece piece = tetrisGame.pieceManager.current;
        for (int[] pos : piece.getCellsAbsolutePosition()) {
            int y = pos[0];
            int x = pos[1];
            if (y >= 0 && y < rows && x >= 0 && x < columns) {
                tetrisGame.gameBoard.images[y][x] = piece.pieceImage;
                tetrisGame.gameBoard.status[y][x] = 1;
            }
        }
    }

    void drawGrid() {
        for (int y = 0; y < tetrisGame.gameBoard.images.length; y++) {
            for (int x = 0; x < tetrisGame.gameBoard.images[y].length; x++) {
                if (tetrisGame.gameBoard.status[y][x] == 0) {
                    drawGameBoardBackground(y, x, positionY + gameBoardY, positionX + gameBoardX);
                } else {
                    image(tetrisGame.gameBoard.images[y][x], positionX + gameBoardX + x * grid, positionY + gameBoardY + y * grid, grid, grid);
                }
            }
        }
    }

    void drawGhostPiece() {
        Piece pretendPiece = new Piece(tetrisGame.pieceManager.current.x, tetrisGame.pieceManager.current.y, tetrisGame.pieceManager.current.shape);
        pretendPiece.rotation = tetrisGame.pieceManager.current.rotation;

        while (tetrisGame.moveHandler.canDrop(pretendPiece)) {
            pretendPiece.y++;
        }

        for (int[] pos : pretendPiece.getCellsAbsolutePosition()) {
            int y = pos[0];
            int x = pos[1];
            if (y >= 0 && y < rows && x >= 0 && x < columns) {
                tint(255, 100);
                image(shape_images.get(pretendPiece.shape), positionX + gameBoardX + x * grid, positionY + gameBoardY + y * grid, grid, grid);
                tint(255, 255);
            }
        }
    }

    void drawScore() {
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 6; x++) {
                image(blockGrayLightBlock, positionX + scoreX + x * grid, positionY + scoreY + y * grid, grid, grid);
            }
        }
        fill(255);
        textSize(round(grid));
        textAlign(LEFT);
        text("V.S SCORE", positionX + scoreX + grid, positionY + scoreY + grid);
        textSize(round(grid * 2));
        textAlign(RIGHT);
        text(tetrisGame.scoreManager.score, positionX + scoreX + 5 * grid, positionY + scoreY + grid * 2.5);
        textAlign(LEFT);
    }

    void drawNextSection() {
        for (int y = 0; y < 16; y++) {
            for (int x = 0; x < 5; x++) {
                drawFunctionBoardBackground(y, x, positionY + nextY, positionX + nextX);
            }
        }
        text("NEXT", positionX + nextX + 0.5f * grid, positionY + nextY + 2 * grid);
        drawNextPieces();
    }

    void drawNextPieces() {
        int count = 0;
        for (Piece nextPiece : tetrisGame.pieceManager.nextQueue) {
            for (int[] cell : nextPiece.getCells()) {
                int y = cell[0];
                int x = cell[1];
                image(nextPiece.pieceImage, positionX + nextPiecePositions.get(count)[1] + x * grid, positionY + nextPiecePositions.get(count)[0] + y * grid, grid, grid);
            }
            count++;
        }
    }

    void drawHoldSection() {
        for (int y = 0; y < 6; y++) {
            for (int x = 0; x < 5; x++) {
                drawFunctionBoardBackground(y, x, positionY + holdY, positionX + holdX);
            }
        }
        text("HOLD", positionX + holdX + 0.5f * grid, positionY + holdY + 2 * grid);
        drawHoldPiece();
    }

    void drawHoldPiece() {
        Piece holdPiece = tetrisGame.pieceManager.hold;
        if (holdPiece == null) return;
        for (int[] cell : holdPiece.getCells()) {
            int y = cell[0];
            int x = cell[1];
            image(holdPiece.pieceImage, holdX + positionX + (1 + x) * grid, holdY + positionY + (2 + y) * grid, grid, grid);
        }
    }

    void render(String name) {
        clearPreviousPiece();
        drawFullBackGround();
        displayCurrentPiece();
        drawGrid();
        drawGhostPiece();
        drawScore();
        text(name, positionX + scoreX + 2 * grid, positionY + (scoreY + grid * 4 + nextY) / 2);
        drawNextSection();
        drawHoldSection();
    }
}
