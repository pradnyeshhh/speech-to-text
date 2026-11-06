"""Hold-to-talk key detection with a Quartz event tap.

A tap sees modifier keys like Fn and Right Option pressed on their own,
which pynput can't do reliably on macOS.
"""

import time

import Quartz

# Key name -> (virtual keycode, device-dependent flag bit that is set while that key is held).
KEYS = {
    "right_option": (61, 0x00000040),
    "right_command": (54, 0x00000010),
    "fn": (63, 0x00800000),
}

# Our own synthetic key events (the Cmd+V paste) carry this tag so the tap ignores them.
SYNTHETIC_EVENT_TAG = 0xB010


class HoldTracker:
    """Turns hotkey transitions into "start", "finish" or "cancel".

    A hold shorter than min_hold_s is cancelled, and so is one where another
    key is pressed while the hotkey is down (a shortcut like Option+E).
    """

    def __init__(self, min_hold_s: float = 0.3):
        self.min_hold_s = min_hold_s
        self._down_at: float | None = None
        self._cancelled = False

    def key_down(self, now: float) -> str | None:
        if self._down_at is not None:
            return None
        self._down_at = now
        self._cancelled = False
        return "start"

    def key_up(self, now: float) -> str | None:
        if self._down_at is None:
            return None
        held = now - self._down_at
        self._down_at = None
        if self._cancelled:
            return None
        return "finish" if held >= self.min_hold_s else "cancel"

    def other_key(self) -> str | None:
        if self._down_at is None or self._cancelled:
            return None
        self._cancelled = True
        return "cancel"


class HotkeyListener:
    """Calls on_start / on_finish / on_cancel from the main thread's run loop."""

    def __init__(self, key: str, on_start, on_finish, on_cancel, min_hold_s: float = 0.3):
        self._keycode, self._flag = KEYS[key]
        self._tracker = HoldTracker(min_hold_s)
        self._handlers = {"start": on_start, "finish": on_finish, "cancel": on_cancel}
        self._tap = None

    def install(self) -> None:
        mask = Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged) | Quartz.CGEventMaskBit(
            Quartz.kCGEventKeyDown
        )
        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            self._on_event,
            None,
        )
        if self._tap is None:
            raise PermissionError("Couldn't watch the keyboard. Is Input Monitoring allowed?")
        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetMain(), source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self._tap, True)

    def _on_event(self, proxy, event_type, event, refcon):
        try:
            self._handle(event_type, event)
        except Exception as exc:  # an exception escaping the callback would kill the tap
            print(f"hotkey error: {exc!r}")
        return event

    def _handle(self, event_type, event) -> None:
        if event_type in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(self._tap, True)
            return
        if Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData) == SYNTHETIC_EVENT_TAG:
            return

        action = None
        if event_type == Quartz.kCGEventFlagsChanged:
            keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            if keycode == self._keycode:
                held = bool(Quartz.CGEventGetFlags(event) & self._flag)
                now = time.monotonic()
                action = self._tracker.key_down(now) if held else self._tracker.key_up(now)
        elif event_type == Quartz.kCGEventKeyDown:
            action = self._tracker.other_key()

        if action:
            self._handlers[action]()
