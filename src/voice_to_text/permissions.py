"""macOS permission checks. While running from a terminal, they apply to the terminal app."""

import Quartz
from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
from AVFoundation import AVAuthorizationStatusDenied, AVCaptureDevice, AVMediaTypeAudio


def missing_permissions() -> list[str]:
    """Ask macOS for whatever is missing and return the names still not granted.

    Microphone isn't asked for here: macOS asks the first time the mic is opened.
    """
    missing = []
    if AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio) == AVAuthorizationStatusDenied:
        missing.append("Microphone")
    if not Quartz.CGPreflightListenEventAccess():
        Quartz.CGRequestListenEventAccess()
        missing.append("Input Monitoring")
    if not AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}):
        missing.append("Accessibility")
    return missing
