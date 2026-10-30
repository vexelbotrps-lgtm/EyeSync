"""Shared Windows identity and icons for every EyeSync window."""
import ctypes
import sys
import app_paths

def set_process_identity():
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "EyeSync.Standalone.Desktop")

def apply_icon(root):
    from PIL import Image, ImageTk
    models = app_paths.APP_DIR / "models"
    # Use the very same ICO used by the executable and Windows shortcut.
    # Do not mix a PNG window icon with a separate native taskbar icon.
    with Image.open(models / "eyesync.ico") as source:
        photo = ImageTk.PhotoImage(source.ico.getimage((64, 64)), master=root)
    root.iconphoto(True, photo)
    if sys.platform == "win32":
        root.iconbitmap(str(models / "eyesync.ico"))
        def native_icons():
            root.update_idletasks()
            user32 = ctypes.windll.user32
            user32.GetParent.argtypes = [ctypes.c_void_p]
            user32.GetParent.restype = ctypes.c_void_p
            user32.LoadImageW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                                         ctypes.c_uint, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_uint]
            user32.LoadImageW.restype = ctypes.c_void_p
            user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                           ctypes.c_size_t, ctypes.c_ssize_t]
            user32.SendMessageW.restype = ctypes.c_ssize_t
            hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
            root._native_icons = []
            for kind, size in ((0, 16), (1, 32)):
                handle = user32.LoadImageW(None, str(models / "eyesync.ico"),
                                          1, size, size, 0x10)
                if not handle:
                    raise ctypes.WinError()
                root._native_icons.append(handle)
                user32.SendMessageW(hwnd, 0x80, kind, handle)  # WM_SETICON
        root.after_idle(native_icons)
    return photo  # Tk needs a retained Python reference.
