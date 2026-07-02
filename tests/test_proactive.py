from nova.server.proactive import ProactiveEngine


def make_engine(**kw):
    defaults = dict(cooldown_s=20.0, talkativeness=0.5, dedupe_window_s=45.0)
    defaults.update(kw)
    return ProactiveEngine(**defaults)


def test_first_event_speaks():
    e = make_engine()
    d = e.on_event("scene_change", now=100.0)
    assert d.speak


def test_cooldown_blocks_then_allows():
    e = make_engine()  # t=0.5 -> effective cooldown = 20s
    assert e.on_event("scene_change", now=100.0).speak
    d = e.on_event("motion_burst", now=105.0)
    assert not d.speak and d.reason == "cooldown"
    assert e.on_event("motion_burst", now=121.0).speak


def test_talkativeness_shrinks_cooldown():
    e = make_engine(talkativeness=1.0)  # effective = 20 * 0.25 = 5s
    assert e.on_event("scene_change", now=100.0).speak
    assert e.on_event("motion_burst", now=106.0).speak


def test_dedupe_same_event_type():
    e = make_engine(cooldown_s=0.1, dedupe_window_s=45.0)
    assert e.on_event("scene_change", now=100.0).speak
    d = e.on_event("scene_change", now=110.0)
    assert not d.speak and d.reason == "dedupe"
    assert e.on_event("scene_change", now=150.0).speak


def test_pause_blocks_and_forced_bypasses_everything():
    e = make_engine()
    assert e.toggle_pause() is True
    assert not e.on_event("scene_change", now=100.0).speak
    assert e.on_event("comment_now", now=100.0, forced=True).speak
    assert e.toggle_pause() is False


def test_forced_bypasses_cooldown_and_dedupe():
    e = make_engine()
    assert e.on_event("scene_change", now=100.0).speak
    assert e.on_event("scene_change", now=101.0, forced=True).speak
