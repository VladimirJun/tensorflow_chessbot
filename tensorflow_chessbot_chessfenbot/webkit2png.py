import sys

from PyQt5.QtCore import QTimer, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication


class WebRenderer(QWebEngineView):
    """Рендеринг веб-страницы и создание скриншота."""

    def __init__(self, url, width=1024, height=768, timeout=5, wait_time=2):
        super().__init__()
        self.url_to_load = url
        self.width = width
        self.height = height
        self.timeout = timeout
        self.wait_time = wait_time

        self.setFixedSize(self.width, self.height)

        # Подключаем сигнал загрузки страницы
        self.page().loadFinished.connect(self.on_load_finished)

        # Запускаем загрузку
        self.load(QUrl(self.url_to_load))

    def on_load_finished(self, success):
        """Коллбэк при завершении загрузки страницы."""
        if not success:
            print("Ошибка загрузки страницы!")
            QApplication.exit(1)

        print("Страница загружена, ждем перед скриншотом...")

        # Ожидание перед рендерингом (для полной загрузки контента)
        QTimer.singleShot(self.wait_time * 1000, self.capture_screenshot)

    def capture_screenshot(self):
        """Создание скриншота после загрузки страницы."""
        self.page().view().grab().save("screenshot.png", "PNG")
        print("Скриншот сохранен как 'screenshot.png'")

        # Завершаем выполнение программы
        QApplication.quit()


def main(url):
    """Запуск QApplication и рендеринг страницы."""
    app = QApplication(sys.argv)
    renderer = WebRenderer(url)
    renderer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    url = "https://www.chess.com/"  # Можно заменить на любой сайт
    main(url)
