class ServerHandler {

    // Handles server communication between Processing and a Python backend
    private PApplet app;
    private String hostName;
    private int port;
    private Process pythonProcess;
    private Client client;
    private boolean isServerReady;
    private Queue<String> pendingMessages;

    // Initializes the server handler with necessary parameters
    ServerHandler(PApplet app, String hostName, int port) {
        this.app = app;
        this.hostName = hostName;
        this.port = port;
        this.isServerReady = false;
        this.pendingMessages = new LinkedList<>();
    }

    // Starts the Python server process and listens for ready signal
    void startServer() {
        try {
            String pythonExecutablePath = app.sketchPath() + "/lib/tetrisAI/bin/python";
            String pythonScriptPath = app.sketchPath() + "/lib/server.py";

            ProcessBuilder processBuilder = new ProcessBuilder(pythonExecutablePath, pythonScriptPath);
            processBuilder.redirectErrorStream(true);
            pythonProcess = processBuilder.start();

            // Creates a background thread to read Python output
            Executors.newSingleThreadExecutor().submit(() -> {
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println("[Python Output]: " + line);
                        if (line.contains("serverReady")) {
                            client = new Client(app, hostName, port);
                            isServerReady = true;
                            while (!pendingMessages.isEmpty()) {
                                client.write(pendingMessages.poll());
                            }
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });

            // Ensures process termination on shutdown
            Runtime.getRuntime().addShutdownHook(new Thread(this::terminateProcess));

            // Registers the dispose method for safe termination
            app.registerMethod("dispose", this);

        } catch (Exception e) {
            terminateProcess();
            e.printStackTrace();
        }
    }

    // Writes a message to the server or queues it if the server is not ready
    void write(String message) {
        if (isServerReady) {
            client.write(message);
        } else {
            pendingMessages.offer(message);
        }
    }

    // Checks if the server has data available
    int available() {
        return isServerReady ? client.available() : 0;
    }

    // Reads a string from the server if available
    String readString() {
        return isServerReady ? client.readString() : "";
    }

    // Ensures process termination when the sketch is disposed
    void dispose() {
        terminateProcess();
    }

    // Stops the server connection and terminates the process
    void stop() {
        if (isServerReady && client != null) {
            client.stop();
        }
    }

    // Terminates the Python process if it's still running
    void terminateProcess() {
        if (pythonProcess != null && pythonProcess.isAlive()) {
            pythonProcess.destroy();
            System.out.println("Python process terminated.");
        }
    }
}
