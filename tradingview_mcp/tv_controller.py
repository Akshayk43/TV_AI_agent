"""TradingView Desktop automation controller.

Uses pyautogui to interact with the TradingView Desktop application:
- Open/focus Pine Editor
- Write Pine Script code
- Compile and add strategy to chart
- Change symbol and timeframe
- Capture screenshots of chart and strategy tester
- Navigate between tabs (Chart, Strategy Tester, etc.)

Platform support: Windows, macOS, Linux (X11).

NOTE: pyautogui requires a display. On headless systems, import will succeed
but methods will raise when called. This is expected — the controller is
designed to run on the user's desktop where TradingView is installed.
"""

import os
import time
import platform
import subprocess
import base64
import io
from datetime import datetime

from PIL import Image

# Lazy-load pyautogui and pyperclip so the module can be imported
# in headless environments (CI, servers) without crashing.
_pyautogui = None
_pyperclip = None


def _get_pyautogui():
    global _pyautogui
    if _pyautogui is None:
        import pyautogui as _mod
        _mod.PAUSE = 0.3
        _mod.FAILSAFE = True
        _pyautogui = _mod
    return _pyautogui


def _get_pyperclip():
    global _pyperclip
    if _pyperclip is None:
        import pyperclip as _mod
        _pyperclip = _mod
    return _pyperclip


PLATFORM = platform.system()  # "Windows", "Darwin", "Linux"

# ── Configurable delays (seconds) ────────────────────────────────────────────
DELAYS = {
    "after_focus": 0.8,
    "after_shortcut": 0.5,
    "after_paste": 0.3,
    "compile_wait": 3.0,
    "backtest_wait": 5.0,
    "symbol_change": 2.0,
    "timeframe_change": 1.5,
    "screenshot_wait": 0.5,
}


class TradingViewController:
    """Controls TradingView Desktop via GUI automation."""

    def __init__(self, delays: dict | None = None):
        self.delays = {**DELAYS, **(delays or {})}
        self._screenshots_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "charts_output"
        )
        os.makedirs(self._screenshots_dir, exist_ok=True)

    # ── Window Management ─────────────────────────────────────────────────

    def focus_tradingview(self) -> bool:
        """Bring TradingView Desktop window to the foreground."""
        time.sleep(0.2)

        if PLATFORM == "Windows":
            try:
                import pygetwindow as gw
                windows = gw.getWindowsWithTitle("TradingView")
                if windows:
                    win = windows[0]
                    if win.isMinimized:
                        win.restore()
                    win.activate()
                    time.sleep(self.delays["after_focus"])
                    return True
            except Exception:
                pass

        elif PLATFORM == "Darwin":
            subprocess.run(
                ["osascript", "-e", 'tell application "TradingView" to activate'],
                capture_output=True,
            )
            time.sleep(self.delays["after_focus"])
            return True

        elif PLATFORM == "Linux":
            result = subprocess.run(
                ["wmctrl", "-a", "TradingView"], capture_output=True,
            )
            if result.returncode != 0:
                subprocess.run(
                    ["xdotool", "search", "--name", "TradingView", "windowactivate"],
                    capture_output=True,
                )
            time.sleep(self.delays["after_focus"])
            return True

        return False

    # ── Pine Editor ───────────────────────────────────────────────────────

    def open_pine_editor(self) -> None:
        """Open the Pine Script Editor panel (Ctrl+/ or Cmd+/)."""
        self.focus_tradingview()
        pag = _get_pyautogui()
        mod = "command" if PLATFORM == "Darwin" else "ctrl"
        pag.hotkey(mod, "/")
        time.sleep(self.delays["after_shortcut"])

    def write_pine_script(self, code: str) -> None:
        """Write Pine Script code into the Pine Editor via clipboard paste."""
        self.open_pine_editor()
        time.sleep(0.5)

        pag = _get_pyautogui()
        mod = "command" if PLATFORM == "Darwin" else "ctrl"

        screen_w, screen_h = pag.size()
        pag.click(screen_w // 2, int(screen_h * 0.75))
        time.sleep(0.3)

        pag.hotkey(mod, "a")
        time.sleep(0.2)

        _get_pyperclip().copy(code)
        pag.hotkey(mod, "v")
        time.sleep(self.delays["after_paste"])

    def compile_and_add_to_chart(self) -> None:
        """Compile Pine Script and add the strategy to the chart."""
        self.focus_tradingview()
        pag = _get_pyautogui()
        mod = "command" if PLATFORM == "Darwin" else "ctrl"

        pag.hotkey(mod, "s")
        time.sleep(1.0)

        try:
            button = pag.locateOnScreen(
                os.path.join(os.path.dirname(__file__), "assets", "add_to_chart.png"),
                confidence=0.7,
            )
            if button:
                pag.click(pag.center(button))
                time.sleep(self.delays["compile_wait"])
                return
        except Exception:
            pass

        pag.hotkey(mod, "Return")
        time.sleep(self.delays["compile_wait"])

    # ── Symbol & Timeframe ────────────────────────────────────────────────

    def change_symbol(self, symbol: str) -> None:
        """Change the chart symbol via the search dialog."""
        self.focus_tradingview()
        pag = _get_pyautogui()
        screen_w, screen_h = pag.size()

        pag.click(int(screen_w * 0.08), int(screen_h * 0.06))
        time.sleep(0.5)

        mod = "command" if PLATFORM == "Darwin" else "ctrl"
        pag.hotkey(mod, "a")
        time.sleep(0.1)
        pag.typewrite(symbol, interval=0.05)
        time.sleep(0.8)
        pag.press("enter")
        time.sleep(self.delays["symbol_change"])

    def change_timeframe(self, timeframe: str) -> None:
        """Change the chart timeframe."""
        self.focus_tradingview()
        pag = _get_pyautogui()

        tf_map = {
            "1m": "1", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "4h": "240",
            "1d": "D", "1w": "W", "1M": "M",
            "1": "1", "5": "5", "15": "15", "60": "60", "240": "240",
            "D": "D", "W": "W", "M": "M",
        }
        tf = tf_map.get(timeframe, timeframe)

        screen_w, screen_h = pag.size()
        pag.click(int(screen_w * 0.15), int(screen_h * 0.06))
        time.sleep(0.5)

        pag.typewrite(tf, interval=0.05)
        time.sleep(0.3)
        pag.press("enter")
        time.sleep(self.delays["timeframe_change"])

    # ── Strategy Tester ───────────────────────────────────────────────────

    def open_strategy_tester(self) -> None:
        """Open the Strategy Tester tab in the bottom panel."""
        self.focus_tradingview()
        pag = _get_pyautogui()

        try:
            tab = pag.locateOnScreen(
                os.path.join(os.path.dirname(__file__), "assets", "strategy_tester_tab.png"),
                confidence=0.7,
            )
            if tab:
                pag.click(pag.center(tab))
                time.sleep(0.5)
                return
        except Exception:
            pass

        screen_w, screen_h = pag.size()
        pag.click(int(screen_w * 0.15), int(screen_h * 0.65))
        time.sleep(0.5)

    def click_overview_tab(self) -> None:
        self._click_strategy_subtab("Overview")

    def click_performance_tab(self) -> None:
        self._click_strategy_subtab("Performance")

    def click_trades_tab(self) -> None:
        self._click_strategy_subtab("Trades")

    def _click_strategy_subtab(self, name: str) -> None:
        pag = _get_pyautogui()
        try:
            tab = pag.locateOnScreen(
                os.path.join(os.path.dirname(__file__), "assets", f"{name.lower()}_tab.png"),
                confidence=0.7,
            )
            if tab:
                pag.click(pag.center(tab))
                time.sleep(0.5)
        except Exception:
            pass

    # ── Screenshots ───────────────────────────────────────────────────────

    def capture_full_screen(self, save: bool = True) -> str:
        """Capture the full screen as base64 PNG."""
        pag = _get_pyautogui()
        time.sleep(self.delays["screenshot_wait"])
        screenshot = pag.screenshot()
        return self._image_to_b64(screenshot, "full_screen" if save else None)

    def capture_chart_area(self, save: bool = True) -> str:
        """Capture the chart area (top ~60% of screen)."""
        pag = _get_pyautogui()
        time.sleep(self.delays["screenshot_wait"])
        screen_w, screen_h = pag.size()
        region = (0, 0, screen_w, int(screen_h * 0.6))
        screenshot = pag.screenshot(region=region)
        return self._image_to_b64(screenshot, "chart" if save else None)

    def capture_strategy_tester(self, save: bool = True) -> str:
        """Capture the Strategy Tester panel (bottom ~40% of screen)."""
        pag = _get_pyautogui()
        self.open_strategy_tester()
        time.sleep(self.delays["screenshot_wait"])
        screen_w, screen_h = pag.size()
        top = int(screen_h * 0.6)
        region = (0, top, screen_w, screen_h - top)
        screenshot = pag.screenshot(region=region)
        return self._image_to_b64(screenshot, "strategy_tester" if save else None)

    def capture_backtest_overview(self, save: bool = True) -> str:
        """Capture the Strategy Tester Overview tab."""
        self.open_strategy_tester()
        self.click_overview_tab()
        time.sleep(0.8)
        return self.capture_strategy_tester(save=save)

    def capture_backtest_performance(self, save: bool = True) -> str:
        """Capture the Performance Summary tab."""
        self.open_strategy_tester()
        self.click_performance_tab()
        time.sleep(0.8)
        return self.capture_strategy_tester(save=save)

    def capture_backtest_trades(self, save: bool = True) -> str:
        """Capture the List of Trades tab."""
        self.open_strategy_tester()
        self.click_trades_tab()
        time.sleep(0.8)
        return self.capture_strategy_tester(save=save)

    # ── Full Workflow Helpers ─────────────────────────────────────────────

    def deploy_strategy(self, pine_script: str, symbol: str | None = None,
                        timeframe: str | None = None) -> None:
        """Full workflow: change symbol/tf -> write script -> compile -> run."""
        self.focus_tradingview()

        if symbol:
            self.change_symbol(symbol)
        if timeframe:
            self.change_timeframe(timeframe)

        self.write_pine_script(pine_script)
        self.compile_and_add_to_chart()
        time.sleep(self.delays["backtest_wait"])

    def get_all_screenshots(self) -> dict[str, str]:
        """Capture chart, overview, performance, and trades screenshots."""
        return {
            "chart": self.capture_chart_area(),
            "overview": self.capture_backtest_overview(),
            "performance": self.capture_backtest_performance(),
            "trades": self.capture_backtest_trades(),
        }

    # ── Scroll & Navigate ─────────────────────────────────────────────────

    def scroll_chart(self, direction: str = "left", amount: int = 5) -> None:
        """Scroll the chart left or right."""
        self.focus_tradingview()
        pag = _get_pyautogui()
        screen_w, screen_h = pag.size()
        pag.click(screen_w // 2, int(screen_h * 0.3))
        time.sleep(0.2)

        key = "left" if direction == "left" else "right"
        for _ in range(amount):
            pag.press(key)
            time.sleep(0.05)

    def zoom_chart(self, direction: str = "in", steps: int = 3) -> None:
        """Zoom the chart in or out."""
        self.focus_tradingview()
        pag = _get_pyautogui()
        screen_w, screen_h = pag.size()
        pag.click(screen_w // 2, int(screen_h * 0.3))
        time.sleep(0.2)

        mod = "command" if PLATFORM == "Darwin" else "ctrl"
        key = "equal" if direction == "in" else "minus"
        for _ in range(steps):
            pag.hotkey(mod, key)
            time.sleep(0.1)

    # ── Internals ─────────────────────────────────────────────────────────

    def _image_to_b64(self, image: Image.Image, name: str | None = None) -> str:
        """Convert PIL Image to base64 PNG, optionally saving to disk."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        if name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self._screenshots_dir, f"{name}_{timestamp}.png")
            image.save(path)

        return base64.b64encode(buf.read()).decode("utf-8")
