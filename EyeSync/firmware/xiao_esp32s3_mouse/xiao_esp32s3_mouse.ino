/*
 * EyeSync - Seeed XIAO ESP32-S3 USB mouse firmware
 * =================================================
 *
 * The XIAO plugs into the computer with one USB-C cable and shows up as TWO
 * USB devices at once:
 *   1. a USB serial port (CDC) that receives commands from live_inference.py
 *   2. an ABSOLUTE-position USB HID mouse that moves the real cursor
 *
 * An absolute mouse (like a touchscreen/tablet) is used instead of the usual
 * relative mouse, so "M,MOVE,960,540" puts the cursor exactly on pixel
 * (960,540) - no drift, no dependence on Windows pointer acceleration.
 * No drivers are needed on Windows, macOS or Linux.
 *
 * Serial protocol (115200 baud, one command per line, '\n' terminated):
 *   M,SCREEN,<W>,<H>   screen resolution in pixels (sent by PC on connect)
 *   M,MOVE,<X>,<Y>     move cursor to absolute screen pixel X,Y
 *   M,CLICK,0,0        left click
 *   M,RCLICK,0,0       right click
 *   M,PING,0,0         keep-alive; replies "EYESYNC,PONG"
 * Replies start with "EYESYNC," and are logged by the PC in eyesync.log.
 *
 * On-board controls:
 *   BOOT button (GPIO0)  press to toggle mouse output on/off (safety switch)
 *   User LED (GPIO21)    solid = connected & enabled, slow blink = disabled,
 *                        off = no data from the PC for 3 s
 *
 * ---------------------------------------------------------------------------
 * Arduino IDE setup (Arduino-ESP32 core 3.x):
 *   1. Boards Manager: install "esp32" by Espressif Systems.
 *   2. Tools > Board            : "XIAO_ESP32S3"
 *   3. Tools > USB Mode         : "USB-OTG (TinyUSB)"      <-- required for HID
 *   4. Tools > USB CDC On Boot  : "Enabled"                <-- serial over USB
 *   5. Upload. If the port disappears, hold BOOT, tap RESET, release BOOT,
 *      then upload again (the board is then in download mode).
 * After uploading, unplug/replug once so the PC enumerates the new HID mouse.
 * ---------------------------------------------------------------------------
 */

#ifndef ARDUINO_USB_MODE
#error "This board has no native USB. Use a XIAO ESP32-S3 (or another ESP32-S2/S3)."
#elif ARDUINO_USB_MODE == 1
#error "Set Tools > USB Mode to 'USB-OTG (TinyUSB)' - HID mouse needs OTG mode."
#endif

#include "USB.h"
#include "USBHID.h"

// ----------------------------------------------------------------------------
// Pins (XIAO ESP32-S3)
// ----------------------------------------------------------------------------
static const int LED_PIN = 21;         // user LED, active LOW
static const int BUTTON_PIN = 0;       // BOOT button, active LOW
static const bool LED_ACTIVE_LOW = true;

// ----------------------------------------------------------------------------
// Absolute mouse HID report descriptor
//   Report: [buttons:1 byte][X:uint16 LE][Y:uint16 LE], X/Y in 0..32767
// ----------------------------------------------------------------------------
static const uint8_t REPORT_ID = 0x0A;
static const uint16_t HID_MAX = 32767;

static const uint8_t ABS_MOUSE_DESCRIPTOR[] = {
  0x05, 0x01,             // Usage Page (Generic Desktop)
  0x09, 0x02,             // Usage (Mouse)
  0xA1, 0x01,             // Collection (Application)
  0x85, REPORT_ID,        //   Report ID
  0x09, 0x01,             //   Usage (Pointer)
  0xA1, 0x00,             //   Collection (Physical)
  0x05, 0x09,             //     Usage Page (Buttons)
  0x19, 0x01,             //     Usage Minimum (1)
  0x29, 0x03,             //     Usage Maximum (3)
  0x15, 0x00,             //     Logical Minimum (0)
  0x25, 0x01,             //     Logical Maximum (1)
  0x95, 0x03,             //     Report Count (3)
  0x75, 0x01,             //     Report Size (1)
  0x81, 0x02,             //     Input (Data, Var, Abs)  - 3 button bits
  0x95, 0x01,             //     Report Count (1)
  0x75, 0x05,             //     Report Size (5)
  0x81, 0x03,             //     Input (Const)           - 5 padding bits
  0x05, 0x01,             //     Usage Page (Generic Desktop)
  0x09, 0x30,             //     Usage (X)
  0x09, 0x31,             //     Usage (Y)
  0x16, 0x00, 0x00,       //     Logical Minimum (0)
  0x26, 0xFF, 0x7F,       //     Logical Maximum (32767)
  0x75, 0x10,             //     Report Size (16)
  0x95, 0x02,             //     Report Count (2)
  0x81, 0x02,             //     Input (Data, Var, Abs)  - absolute X, Y
  0xC0,                   //   End Collection
  0xC0                    // End Collection
};

USBHID HID;

class AbsoluteMouse : public USBHIDDevice {
public:
  AbsoluteMouse() {
    static bool registered = false;
    if (!registered) {
      registered = true;
      HID.addDevice(this, sizeof(ABS_MOUSE_DESCRIPTOR));
    }
  }

  void begin() { HID.begin(); }

  uint16_t _onGetDescriptor(uint8_t *buffer) override {
    memcpy(buffer, ABS_MOUSE_DESCRIPTOR, sizeof(ABS_MOUSE_DESCRIPTOR));
    return sizeof(ABS_MOUSE_DESCRIPTOR);
  }

  bool send(uint8_t buttons, uint16_t x, uint16_t y) {
    if (!HID.ready()) return false;      // not enumerated / host asleep
    uint8_t report[5] = {buttons, (uint8_t)(x & 0xFF), (uint8_t)(x >> 8),
                         (uint8_t)(y & 0xFF), (uint8_t)(y >> 8)};
    return HID.SendReport(REPORT_ID, report, sizeof(report));
  }
};

AbsoluteMouse Mouse;   // global so it registers before USB starts

// ----------------------------------------------------------------------------
// State
// ----------------------------------------------------------------------------
static long screenW = 1920, screenH = 1080;   // updated by M,SCREEN
static uint16_t curX = HID_MAX / 2, curY = HID_MAX / 2;
static bool enabled = true;
static unsigned long lastRxMs = 0;

static char lineBuf[64];
static size_t lineLen = 0;
static bool lineOverflow = false;

static void setLed(bool on) {
  digitalWrite(LED_PIN, (on ^ LED_ACTIVE_LOW) ? HIGH : LOW);
}

static uint16_t toHid(long px, long size) {
  if (size < 2) return 0;
  if (px < 0) px = 0;
  if (px > size - 1) px = size - 1;
  return (uint16_t)((px * (long)HID_MAX) / (size - 1));
}

// ----------------------------------------------------------------------------
// Command handling
// ----------------------------------------------------------------------------
static void handleLine(char *line) {
  char cmd[12];
  long a = 0, b = 0;
  // Expected: M,<CMD>,<A>,<B>
  if (sscanf(line, "M,%11[^,],%ld,%ld", cmd, &a, &b) != 3) {
    Serial.printf("EYESYNC,ERR,bad command: %s\n", line);
    return;
  }

  if (strcmp(cmd, "MOVE") == 0) {
    curX = toHid(a, screenW);
    curY = toHid(b, screenH);
    if (enabled) Mouse.send(0, curX, curY);
  } else if (strcmp(cmd, "CLICK") == 0) {
    if (enabled) {
      Mouse.send(0x01, curX, curY);   // left button down at current position
      delay(15);
      Mouse.send(0x00, curX, curY);   // and up
    }
  } else if (strcmp(cmd, "RCLICK") == 0) {
    if (enabled) {
      Mouse.send(0x02, curX, curY);   // right button down at current position
      delay(15);
      Mouse.send(0x00, curX, curY);   // and up
    }
  } else if (strcmp(cmd, "SCREEN") == 0) {
    if (a >= 2 && b >= 2 && a <= 16384 && b <= 16384) {
      screenW = a;
      screenH = b;
      Serial.printf("EYESYNC,SCREEN,%ld,%ld\n", screenW, screenH);
    }
  } else if (strcmp(cmd, "PING") == 0) {
    Serial.println("EYESYNC,PONG");
  } else {
    Serial.printf("EYESYNC,ERR,unknown command: %s\n", cmd);
  }
}

static void pollSerial() {
  // Read everything available; assemble lines in a fixed buffer (no String
  // objects -> no heap fragmentation during days of continuous running).
  while (Serial.available() > 0) {
    int c = Serial.read();
    if (c < 0) break;
    lastRxMs = millis();
    if (c == '\r') continue;
    if (c == '\n') {
      if (!lineOverflow && lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
      }
      lineLen = 0;
      lineOverflow = false;
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = (char)c;
    } else {
      lineOverflow = true;   // drop over-long garbage lines entirely
    }
  }
}

static void pollButton() {
  // Debounced BOOT button: each press toggles mouse output.
  static int lastReading = HIGH, stableState = HIGH;
  static unsigned long changedAt = 0;
  int reading = digitalRead(BUTTON_PIN);
  if (reading != lastReading) {
    changedAt = millis();
    lastReading = reading;
  }
  if (millis() - changedAt > 30 && reading != stableState) {
    stableState = reading;
    if (stableState == LOW) {
      enabled = !enabled;
      Serial.printf("EYESYNC,ENABLED,%d\n", enabled ? 1 : 0);
    }
  }
}

static void updateLed() {
  unsigned long now = millis();
  bool hostActive = lastRxMs != 0 && now - lastRxMs < 3000;
  if (!enabled) setLed((now / 500) % 2);   // slow blink = disabled
  else setLed(hostActive);                 // solid = receiving commands
}

// ----------------------------------------------------------------------------
void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  setLed(false);

  Serial.begin(115200);   // USB CDC: baud rate is nominal, any value works
  USB.productName("EyeSync Mouse");
  USB.manufacturerName("EyeSync");
  Mouse.begin();
  USB.begin();

  Serial.println("EYESYNC,READY,XIAO-ESP32S3");
}

void loop() {
  pollSerial();
  pollButton();
  updateLed();
  delay(1);   // yield to the USB stack / watchdog
}
