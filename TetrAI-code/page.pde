abstract class Page {

    // Constructor that automatically calls init()
    Page() {
        init();
    }

    // Initializes the page, only executed once
    abstract void init();

    // Renders the page content
    abstract void render();

    // Handles click events
    abstract void onClick();
    
    // Handles keyPress event
    abstract void onKeyPressed(int key, int keyCode);
}

class PageManager {
    private Page currentPage;

    // 切換頁面
    void switchPage(Page newPage) {
        if (currentPage != null) {
            currentPage = null;  // 清除記憶體
        }
        currentPage = newPage;
    }

    // 渲染頁面
    void renderCurrentPage() {
        if (currentPage != null) {
            currentPage.render();
        }
    }

    // 點擊處理
    void onClickCurrentPage() {
        if (currentPage != null) {
            currentPage.onClick();
        }
    }
    
    // Handle keyPressed
    void onKeyPressCurrentPage(int key, int keyCode) {
        if (currentPage != null) {
            currentPage.onKeyPressed(key, keyCode);
        }
    }
}
