class CollisionManager {

    GameBoard gameBoard;
    PieceManager pieceManager;

    CollisionManager(GameBoard gameBoard, PieceManager pieceManager) {
        this.gameBoard = gameBoard;
        this.pieceManager = pieceManager;
    }

    // Determines if the piece can move left
    boolean canMoveLeft() {
        Piece piece = pieceManager.current;
        ArrayList<int[]> positions = piece.getCellsAbsolutePosition();
        for (int[] pos : positions) {
            int x = pos[1];
            int y = pos[0];
            if (x == 0 || y < 0 || gameBoard.status[y][x - 1] == 2) {
                return false;
            }
        }
        return true;
    }

    // Determines if the piece can move right
    boolean canMoveRight() {
        Piece piece = pieceManager.current;
        ArrayList<int[]> positions = piece.getCellsAbsolutePosition();
        for (int[] pos : positions) {
            int x = pos[1];
            int y = pos[0];
            if (x == 9 || y < 0 || gameBoard.status[y][x + 1] == 2) {
                return false;
            }
        }
        return true;
    }
    
    // Determines if the piece can drop
    boolean canDrop(Piece piece) {
        ArrayList<int[]> positions = piece.getCellsAbsolutePosition();
        for (int[] pos : positions) {
            int y = pos[0];
            int x = pos[1];
            if (y >= 0 && x >= 0 && (y + 1 == 20 || gameBoard.status[y + 1][x] == 2)) {
                return false;
            }
        }
        return true;
    }
    
    // Overload canDrop
    boolean canDrop() { return canDrop(pieceManager.current); }

    // Determines if the piece can rotate
    boolean canRotate() {
        Piece piece = pieceManager.current;
        ArrayList<int[]> positions = piece.getCellsAbsolutePosition();
        for (int[] pos : positions) {
            int x = pos[1];
            int y = pos[0];
            if (x < 0 || x >= 10 || y >= 20 || y < 0 || gameBoard.status[y][x] == 2) {
                return false;
            }
        }
        return true;
    }
}

class MoveHandler extends CollisionManager {

    MoveHandler(GameBoard gameBoard, PieceManager pieceManager) {
        super(gameBoard, pieceManager);
    }

    // Moves the piece left by one column if possible
    void moveLeft() {
        if (canMoveLeft()) {
            pieceManager.current.x -= 1;
        }
    }

    // Moves the piece right by one column if possible
    void moveRight() {
        if (canMoveRight()) {
            pieceManager.current.x += 1;
        }
    }

    // Drops the piece by one row if possible, otherwise fixes it
    void drop() {
        if (canDrop()) {
            pieceManager.current.y += 1;
        } else {
            fixPiece();
        }
    }

    // Instantly drops the piece to the lowest possible row
    void instantDrop() {
        while (canDrop()) {
            pieceManager.current.y += 1;
        }
        fixPiece();
    }

    // Rotates the piece clockwise or counterclockwise if possible
    void rotate(boolean counterClockWise) {
        if (counterClockWise) {
            pieceManager.current.rotation -= 1;
        } else {
            pieceManager.current.rotation += 1;
        }
        if (!canRotate()) {
            if (counterClockWise) {
                pieceManager.current.rotation += 1;
            } else {
                pieceManager.current.rotation -= 1;
            }
        }
    }

    // Fixes the landed piece on the game board by marking its cells as occupied
    void fixPiece() {
        pieceManager.current.is_fixed = true;
        ArrayList<int[]> positions = pieceManager.current.getCellsAbsolutePosition();

        for (int[] pos : positions) {
            int y = pos[0];
            int x = pos[1];
            gameBoard.status[y][x] = 2;  // Mark as fixed
            gameBoard.images[y][x] = pieceManager.current.pieceImage;
        }
    }
}
