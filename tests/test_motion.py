from nova.client.motion import MotionGate


def test_still_by_default():
    g = MotionGate()
    assert not g.is_motion(ts=100.0)


def test_motion_after_burst_of_events():
    g = MotionGate(on_events=3, on_window_s=30.0, off_silence_s=60.0)
    for t in (100.0, 105.0, 110.0):
        g.note_event(t)
    assert g.is_motion(ts=111.0)


def test_sparse_events_stay_still():
    g = MotionGate(on_events=3, on_window_s=30.0)
    for t in (100.0, 140.0, 180.0):   # реже окна
        g.note_event(t)
    assert not g.is_motion(ts=181.0)


def test_motion_decays_after_silence():
    g = MotionGate(on_events=3, on_window_s=30.0, off_silence_s=60.0)
    for t in (100.0, 101.0, 102.0):
        g.note_event(t)
    assert g.is_motion(ts=110.0)
    assert g.is_motion(ts=161.0)          # 59с тишины — ещё смотрим
    assert not g.is_motion(ts=163.0)      # 61с — расслабилась


def test_cinema_forces_motion_regardless():
    g = MotionGate()
    g.set_cinema(True)
    assert g.is_motion(ts=100.0)
    assert g.cinema
    g.set_cinema(False)
    assert not g.is_motion(ts=100.0)
