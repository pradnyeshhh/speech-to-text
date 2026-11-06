from voice_to_text.hotkey import HoldTracker


def test_hold_then_release_finishes():
    tracker = HoldTracker(min_hold_s=0.3)
    assert tracker.key_down(0.0) == "start"
    assert tracker.key_up(1.2) == "finish"


def test_short_tap_is_cancelled():
    tracker = HoldTracker(min_hold_s=0.3)
    assert tracker.key_down(0.0) == "start"
    assert tracker.key_up(0.1) == "cancel"


def test_other_key_during_hold_cancels_once():
    tracker = HoldTracker(min_hold_s=0.3)
    tracker.key_down(0.0)
    assert tracker.other_key() == "cancel"
    assert tracker.other_key() is None
    assert tracker.key_up(2.0) is None


def test_repeated_key_down_is_ignored():
    tracker = HoldTracker(min_hold_s=0.3)
    assert tracker.key_down(0.0) == "start"
    assert tracker.key_down(0.5) is None
    assert tracker.key_up(1.0) == "finish"


def test_keys_while_idle_do_nothing():
    tracker = HoldTracker()
    assert tracker.other_key() is None
    assert tracker.key_up(1.0) is None


def test_next_hold_works_after_a_cancel():
    tracker = HoldTracker(min_hold_s=0.3)
    tracker.key_down(0.0)
    tracker.other_key()
    tracker.key_up(0.5)
    assert tracker.key_down(1.0) == "start"
    assert tracker.key_up(2.0) == "finish"
