class Piece {
    int x;  // Center position (row)
    int y;  // Center position (column)
    int rotation;
    PImage pieceImage;  // Corresponding image
    boolean is_fixed;
    String shape;

    // Constructor to initialize the piece
    Piece(int row, int column, String shape) {
        this.x = row;
        this.y = column;
        this.shape = shape;
        this.rotation = 0;
        this.pieceImage = getImage(shape);
        this.is_fixed = false;
    }

    // Retrieves the corresponding image based on the shape
    PImage getImage(String shape) {
        return shape_images.get(shape);
    }

    // Retrieves the current cell configuration based on rotation
    int[][] getCells() {
        ArrayList<int[][]> shapeArray = shapes.get(shape);  // Get shapes from config
        // handle negative shit
        int tmp = this.rotation % shapeArray.size();
        int index = tmp >= 0 ? tmp : shapeArray.size() + tmp;
        return shapeArray.get(index);
    }
    
    // Retrieves the absolute positions of all cells in the piece
    ArrayList<int[]> getCellsAbsolutePosition() {
        ArrayList<int[]> positions = new ArrayList<>();
        int[][] cells = getCells(); // Get shape definition y, x
        for (int[] cell : cells) {
            int x = cell[1];
            int y = cell[0];
            positions.add(new int[]{ y + this.y, x + this.x });
        }
        return positions;
    }
}

class PieceManager {
    Piece current;
    Piece next;
    Piece hold;
    Queue<Piece> nextQueue = new LinkedList<>();
    Queue<String> bag = new LinkedList<>();
    boolean holdLocked;
    Random rand = new Random();
    
    PieceManager() {
        bag.clear();
        nextQueue.clear();
        current = getRandomPiece();
        for (int i = 0; i < 3; i++) nextQueue.offer(getRandomPiece());
        next = new Piece(3, 0, nextQueue.element().shape);
        holdLocked = false;
    }
    
    // Randomly generates a new piece
    Piece getRandomPiece() {
        String[] shapeKeys = shapes.keySet().toArray(new String[0]);
        if (bag.isEmpty()) {
            shuffleBag(shapeKeys);
            for (String shape : shapeKeys) {
                bag.offer(shape);
            }
        }
        String shape = bag.poll();
        return new Piece(4, 0, shape);  // Initial position
    }

    // Shuffles the bag using Fisher-Yates shuffle
    private void shuffleBag(String[] shapeKeys) {
        for (int i = shapeKeys.length - 1; i > 0; i--) {
            int index = rand.nextInt(i + 1);
            String temp = shapeKeys[index];
            shapeKeys[index] = shapeKeys[i];
            shapeKeys[i] = temp;
        }
    }

    // Holds the current piece
    void hold() {
        if (!holdLocked) {
            if (hold == null) {
                hold = new Piece(5, 0, current.shape);
                current = nextQueue.poll();
                nextQueue.offer(getRandomPiece());
            } else {
                String shape = current.shape;
                current = new Piece(5, 0, hold.shape);
                hold = new Piece(5, 0, shape);
            }
            holdLocked = true;
        }
    }

    // Updates the piece when fixed
    void onFixed() {
        holdLocked = false;
        current = nextQueue.poll();
        nextQueue.offer(getRandomPiece());
    }
}
