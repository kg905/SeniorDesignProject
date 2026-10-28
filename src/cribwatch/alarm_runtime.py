"""Portable receiver: no TensorFlow, MediaPipe, NumPy, sklearn, or camera dependency."""

import argparse
import json
import threading
import time

from .transport import ReceiverState, get_key, make_server


def receiver(args):
    state = ReceiverState(get_key(), timeout=args.timeout)
    if getattr(args, "protocol", "http") == "tcp":
        from .tcp_alarm import make_tcp_server
        server = make_tcp_server(args.host, args.port, state)
    else:
        server = make_server(args.host, args.port, state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    buzzer = button = None
    muted_until = [0.0]
    if args.gpio is not None:
        from gpiozero import DigitalOutputDevice
        buzzer = DigitalOutputDevice(args.gpio, initial_value=False)
    if args.mute_button is not None:
        from gpiozero import Button
        button = Button(args.mute_button, pull_up=True, bounce_time=.1)
        # Temporary local silence never clears the camera's latched alarm.
        button.when_pressed = lambda: muted_until.__setitem__(0, time.monotonic() + 10)
    worker.start()
    previous = None
    try:
        while True:
            status = state.snapshot()
            audible = status["audible"] and time.monotonic() >= muted_until[0]
            # Continuous tone for baby alert; pulsed tone for monitor/link fault.
            on = audible and (status["baby_alarm"] or time.monotonic() % 1 < .35)
            if buzzer:
                buzzer.value = on
            if status != previous:
                print(json.dumps(status), flush=True)
                previous = status
            time.sleep(.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        if buzzer:
            buzzer.off()
            buzzer.close()
        if button:
            button.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Portable alarm receiver, TCP 5000")
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--timeout', type=float, default=5.)
    parser.add_argument('--gpio', type=int)
    parser.add_argument('--mute-button', type=int)
    args = parser.parse_args(argv)
    args.protocol = 'tcp'
    receiver(args)


if __name__ == '__main__':
    main()
