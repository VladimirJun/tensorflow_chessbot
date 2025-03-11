import sys
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebEngineCore import QWebEngineSettings


class ChessScreenshotServer:
    """Класс для создания скриншотов шахматных досок с Lichess"""

    def __init__(self, url=None, output_filename=None):
        self.url = url
        self.output_filename = output_filename
        self.app = QApplication(sys.argv)
        self.view = QWebEngineView()

        # Включаем поддержку JavaScript
        settings = self.view.settings()
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)

    def render_screenshot(self):
        """Создает скриншот после загрузки страницы"""
        def capture():
            self.view.grab().save(self.output_filename, "PNG")
            print(f"Скриншот сохранен в {self.output_filename}")
            self.app.quit()

        # Запускаем таймер, чтобы дождаться загрузки страницы перед скриншотом
        QTimer.singleShot(2000, capture)

    def take_screenshot(self, url=None, output_filename=None):
        """Загружает страницу и сохраняет скриншот"""
        if url:
            self.url = url
        if output_filename:
            self.output_filename = output_filename

        self.view.loadFinished.connect(self.render_screenshot)
        self.view.load(self.url)
        sys.exit(self.app.exec_())

    def take_chess_screenshot(self, fen_string, output_filename):
        """Создает скриншот шахматной доски на Lichess по FEN"""
        url_template = f"https://lichess.org/editor/{fen_string}"
        self.take_screenshot(url_template, output_filename)


if __name__ == "__main__":
    chess_scraper = ChessScreenshotServer()
    chess_scraper.take_chess_screenshot("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1", "chessboard.png")
