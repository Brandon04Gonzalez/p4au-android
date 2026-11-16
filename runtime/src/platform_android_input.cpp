#include <cstdlib>
// Android host (NativeActivity): memory (mmap at fixed addresses), pthreads, the app's window (ANativeWindow),
// game controllers + touch, AAudio, the Vulkan loader and android surface, a signal crash handler and
// android_main. Game data lives in the app's external files dir (see docs/ANDROID.md):
//   <files>/analysis/{main,sdk}.bin   module images (tools/nso2bin.py)
//   <files>/extract/romfs/...         the RomFS
//   <files>/user/...                  saves
//   <files>/config.txt                optional: resolution=WxH
#include <aaudio/AAudio.h>
#include <android/log.h>
#include <android/native_window.h>
#include <android/window.h>
#include <android_native_app_glue.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <jni.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>

#include <nx/game.h>
#include <nx/runtime.h>

#include "kernel.h"
#include "platform.h"
#include "platform_android.h"
#include "touch_layout.h"

// Android input: game controllers and touch (split from platform_android.cpp).
namespace nx::plat {

// ---- input: game controllers + touch --------------------------------------------------------------------------
// Touch: left half = virtual stick (relative to where the finger went down); right half = four buttons in a
// diamond around the thumb (Y attack up-left, X special up-right, B dash down-left, A interact down-right), top
// corners L / R, top centre +. Games with a drawn layout (touch_layout.cpp) use its buttons instead, and may make the
// rest of the right half a second stick (P5R: the camera).

namespace {
std::mutex g_in;
PadState g_pads[2];        // physical controller slots (P1, P2) keyed by Android InputDevice id
PadState g_kbd_buttons;    // non-gamepad keyboard / system buttons (routed to P1)
int32_t g_pad_dev[2] = {-1, -1};
JavaVM* g_vm = nullptr;
jclass g_input_dev_cls = nullptr;
jmethodID g_get_dev_mid = nullptr;

void init_jni(android_app* app) {
    if (g_get_dev_mid || !app || !app->activity || !app->activity->vm) return;
    g_vm = app->activity->vm;
    JNIEnv* env = nullptr;
    bool attached = false;
    jint st = g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (st == JNI_EDETACHED) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) return;
        attached = true;
    } else if (st != JNI_OK || !env) {
        return;
    }
    jclass cls = env->FindClass("android/view/InputDevice");
    if (cls) {
        g_input_dev_cls = static_cast<jclass>(env->NewGlobalRef(cls));
        g_get_dev_mid = env->GetStaticMethodID(g_input_dev_cls, "getDevice", "(I)Landroid/view/InputDevice;");
        env->DeleteLocalRef(cls);
    }
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (attached) g_vm->DetachCurrentThread();
}

bool device_connected(int32_t dev_id) {
    if (dev_id < 0 || !g_vm || !g_input_dev_cls || !g_get_dev_mid) return dev_id >= 0;
    JNIEnv* env = nullptr;
    bool attached = false;
    jint st = g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (st == JNI_EDETACHED) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK || !env) return true;
        attached = true;
    } else if (st != JNI_OK || !env) {
        return true;
    }
    jobject dev = env->CallStaticObjectMethod(g_input_dev_cls, g_get_dev_mid, static_cast<jint>(dev_id));
    bool ok = dev != nullptr;
    if (dev) env->DeleteLocalRef(dev);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        ok = true;
    }
    if (attached) g_vm->DetachCurrentThread();
    return ok;
}

int slot_for_device(int32_t dev, bool can_claim) {
    if (dev == g_pad_dev[0]) return 0;
    if (dev == g_pad_dev[1]) return 1;
    if (!can_claim || dev < 0) return -1;
    if (g_pad_dev[0] < 0 || !device_connected(g_pad_dev[0])) {
        g_pad_dev[0] = dev;
        g_pads[0] = PadState{};
        log("hid: android pad 1 claimed device %d", dev);
        return 0;
    }
    if (g_pad_dev[1] < 0 || !device_connected(g_pad_dev[1])) {
        g_pad_dev[1] = dev;
        g_pads[1] = PadState{};
        log("hid: android pad 2 claimed device %d (pad 1 is %d)", dev, g_pad_dev[0]);
        return 1;
    }
    return -1;
}
enum Stick : uint8_t { kNoStick, kLeftStick, kRightStick };
struct Touch {
    int32_t id;
    float x0, y0, x, y;
    Stick stick;
    uint64_t button;
    uint64_t down_ns;
    bool dpad;
};
std::vector<Touch> g_touches;
// A quick tap still reaches the game: a button let go sooner than kMinHoldNs after it went down stays held until
// then, unless tapped again (so fast button mashing always registers distinct presses).
struct Latch {
    uint64_t button, until_ns;
};
std::vector<Latch> g_latched;
constexpr uint64_t kMinHoldNs = 25'000'000;

void release_touch(const Touch& t, uint64_t now) {
    if (t.button && now < t.down_ns + kMinHoldNs) g_latched.push_back({t.button, t.down_ns + kMinHoldNs});
}
float g_w = 1, g_h = 1;
bool g_gamepad = false;  // the last input came from a controller: the drawn touch controls hide
uint64_t g_last_touch_ns = 0;  // the drawn controls dim after kTouchIdleNs without a touch
constexpr uint64_t kTouchIdleNs = 8'000'000'000;

bool gamepad_swap_abxy() {
    static const bool swap = [] {
        const char* e = std::getenv("NX_GAMEPAD_LAYOUT");
        return e && std::strcmp(e, "1") == 0;
    }();
    return swap;
}

float stick_sens() {
    static const float s = [] {
        const char* e = std::getenv("NX_STICK_SENSITIVITY");
        int v = e ? std::atoi(e) : 100;
        if (v < 25) v = 25;
        if (v > 300) v = 300;
        return float(v) / 100.0f;
    }();
    return s;
}

float camera_sens() {
    static const float s = [] {
        const char* e = std::getenv("NX_CAMERA_SENSITIVITY");
        int v = e ? std::atoi(e) : 100;
        if (v < 25) v = 25;
        if (v > 300) v = 300;
        return float(v) / 100.0f;
    }();
    return s;
}

uint64_t key_to_button(int32_t k) {
    if (gamepad_swap_abxy()) {
        switch (k) {
            case AKEYCODE_BUTTON_A:
            case AKEYCODE_BUTTON_3: return kA;  // bottom face button (Cross / A)
            case AKEYCODE_BUTTON_B:
            case AKEYCODE_BUTTON_2: return kB;  // right face button (Circle / B)
            case AKEYCODE_BUTTON_X:
            case AKEYCODE_BUTTON_4: return kX;  // left face button (Square / X)
            case AKEYCODE_BUTTON_Y:
            case AKEYCODE_BUTTON_1: return kY;  // top face button (Triangle / Y)
            default: break;
        }
    } else {
        switch (k) {
            case AKEYCODE_BUTTON_A:
            case AKEYCODE_BUTTON_3: return kB;  // bottom face button = Nintendo B
            case AKEYCODE_BUTTON_B:
            case AKEYCODE_BUTTON_2: return kA;  // right face button = Nintendo A
            case AKEYCODE_BUTTON_X:
            case AKEYCODE_BUTTON_4: return kY;  // left face button = Nintendo Y
            case AKEYCODE_BUTTON_Y:
            case AKEYCODE_BUTTON_1: return kX;  // top face button = Nintendo X
            default: break;
        }
    }
    switch (k) {
        case AKEYCODE_BUTTON_L1:
        case AKEYCODE_BUTTON_7: return kL;
        case AKEYCODE_BUTTON_R1:
        case AKEYCODE_BUTTON_8: return kR;
        case AKEYCODE_BUTTON_L2:
        case AKEYCODE_BUTTON_5:
        case AKEYCODE_BUTTON_C: return kZL;
        case AKEYCODE_BUTTON_R2:
        case AKEYCODE_BUTTON_6:
        case AKEYCODE_BUTTON_Z: return kZR;
        case AKEYCODE_BUTTON_START:
        case AKEYCODE_BUTTON_10:
        case AKEYCODE_BUTTON_MODE: return kPlus;
        case AKEYCODE_BUTTON_SELECT:
        case AKEYCODE_BUTTON_9: return kMinus;
        case AKEYCODE_BUTTON_THUMBL:
        case AKEYCODE_BUTTON_11: return kStickL;
        case AKEYCODE_BUTTON_THUMBR:
        case AKEYCODE_BUTTON_12: return kStickR;
        case AKEYCODE_DPAD_LEFT: return kLeft;
        case AKEYCODE_DPAD_UP: return kUp;
        case AKEYCODE_DPAD_RIGHT: return kRight;
        case AKEYCODE_DPAD_DOWN: return kDown;
        case AKEYCODE_BACK: return kMinus;
        // A keyboard: Enter / Space confirm (A), Escape goes back (B). (Some tablets keep BUTTON_B for their stylus.)
        case AKEYCODE_ENTER: case AKEYCODE_NUMPAD_ENTER: case AKEYCODE_SPACE: case AKEYCODE_DPAD_CENTER: return kA;
        case AKEYCODE_ESCAPE: return kB;
        default: return 0;
    }
}

uint64_t touch_button(float x, float y) {
    float nx_ = x / g_w, ny = y / g_h;
    if (ny < 0.2f) return nx_ < 0.6f ? (nx_ < 0.45f ? kL : kPlus) : (nx_ > 0.85f ? kR : kZR);
    bool up = ny < 0.6f, right = nx_ > 0.8f;
    if (up) return right ? kX : kY;
    return right ? kA : kB;
}

}  // namespace

int32_t on_input(android_app* app, AInputEvent* e) {
    std::lock_guard lk(g_in);
    init_jni(app);
    int32_t type = AInputEvent_getType(e);
    int32_t src = AInputEvent_getSource(e);
    int32_t dev = AInputEvent_getDeviceId(e);
    if (type == AINPUT_EVENT_TYPE_KEY) {
        uint64_t b = key_to_button(AKeyEvent_getKeyCode(e));
        if (!b) return 0;
        int32_t action = AKeyEvent_getAction(e);
        const bool is_pad_src =
            (src & (AINPUT_SOURCE_GAMEPAD | AINPUT_SOURCE_JOYSTICK | AINPUT_SOURCE_DPAD)) != 0;
        if (!is_pad_src) {
            if (action == AKEY_EVENT_ACTION_DOWN)
                g_kbd_buttons.buttons |= b;
            else if (action == AKEY_EVENT_ACTION_UP)
                g_kbd_buttons.buttons &= ~b;
            return 1;
        }
        int slot = slot_for_device(dev, action == AKEY_EVENT_ACTION_DOWN);
        if (slot < 0) return 0;
        g_gamepad = true;
        if (action == AKEY_EVENT_ACTION_DOWN)
            g_pads[slot].buttons |= b;
        else if (action == AKEY_EVENT_ACTION_UP)
            g_pads[slot].buttons &= ~b;
        return 1;
    }
    if (type != AINPUT_EVENT_TYPE_MOTION) return 0;
    if ((src & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK) {
        // Radial 2D deadzone so circular physical sticks still reach 100% magnitude on diagonals.
        auto stick_axes = [&](int32_t ax, int32_t ay, int32_t& out_x, int32_t& out_y) {
            float x = AMotionEvent_getAxisValue(e, ax, 0);
            float y = -AMotionEvent_getAxisValue(e, ay, 0);
            constexpr float kDeadzone = 0.12f;
            constexpr float kOuter = 0.95f;
            float mag = std::hypot(x, y);
            if (mag <= kDeadzone) {
                out_x = 0;
                out_y = 0;
                return;
            }
            float scaled = std::min(1.0f, (mag - kDeadzone) / (kOuter - kDeadzone));
            float inv = scaled / mag;
            out_x = int32_t(std::clamp(x * inv, -1.f, 1.f) * kStickMax);
            out_y = int32_t(std::clamp(y * inv, -1.f, 1.f) * kStickMax);
        };
        int32_t lx = 0, ly = 0, rx = 0, ry = 0;
        stick_axes(AMOTION_EVENT_AXIS_X, AMOTION_EVENT_AXIS_Y, lx, ly);
        stick_axes(AMOTION_EVENT_AXIS_Z, AMOTION_EVENT_AXIS_RZ, rx, ry);
        if (!rx && !ry) stick_axes(AMOTION_EVENT_AXIS_RX, AMOTION_EVENT_AXIS_RY, rx, ry);
        float hx = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_X, 0);
        float hy = AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_HAT_Y, 0);
        uint64_t motion_btns = 0;
        if (hx < -0.5f) motion_btns |= kLeft;
        if (hx > 0.5f) motion_btns |= kRight;
        if (hy < -0.5f) motion_btns |= kUp;
        if (hy > 0.5f) motion_btns |= kDown;
        if (AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_LTRIGGER, 0) > 0.4f ||
            AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_BRAKE, 0) > 0.4f)
            motion_btns |= kZL;
        if (AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_RTRIGGER, 0) > 0.4f ||
            AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_GAS, 0) > 0.4f)
            motion_btns |= kZR;
        const bool active = (lx || ly || rx || ry || motion_btns != 0);
        int slot = slot_for_device(dev, active);
        if (slot < 0) return 0;
        g_gamepad = true;
        PadState& dst = g_pads[slot];
        dst.lx = lx;
        dst.ly = ly;
        dst.rx = rx;
        dst.ry = ry;
        dst.buttons = (dst.buttons & ~(kLeft | kRight | kUp | kDown | kZL | kZR)) | motion_btns;
        return 1;
    }
    if ((src & AINPUT_SOURCE_TOUCHSCREEN) != AINPUT_SOURCE_TOUCHSCREEN) return 0;
    if (!touch_enabled()) return 0;  // NX_TOUCH_SHOW=0
    int32_t action = AMotionEvent_getAction(e);
    int32_t kind = action & AMOTION_EVENT_ACTION_MASK;
    size_t index = size_t((action & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT);
    const uint64_t now = monotonic_ns();
    g_last_touch_ns = now;
    if (ANativeWindow* w = android_window()) {
        g_w = float(ANativeWindow_getWidth(w));
        g_h = float(ANativeWindow_getHeight(w));
    }
    if (kind == AMOTION_EVENT_ACTION_DOWN || kind == AMOTION_EVENT_ACTION_POINTER_DOWN) {
        float x = AMotionEvent_getX(e, index), y = AMotionEvent_getY(e, index);
        g_gamepad = false;
        size_t drawn = 0;
        touch_buttons(drawn);
        if (drawn) {  // a drawn layout (touch_layout.cpp): its buttons first, the rest of the left half is the stick
            uint64_t b = touch_hit(x / g_w, y / g_h, g_w / g_h);
            if (b) {
                g_latched.erase(std::remove_if(g_latched.begin(), g_latched.end(),
                                               [&](const Latch& l) { return (l.button & b) != 0; }),
                                g_latched.end());
            }
            bool dpad = (b & (kUp | kDown | kLeft | kRight)) != 0;
            bool left = !b && touch_stick_side(x / g_w);
            Stick stick = b ? kNoStick : left ? kLeftStick : touch_right_stick() ? kRightStick : kNoStick;
            if (b || stick) g_touches.push_back({AMotionEvent_getPointerId(e, index), x, y, x, y, stick, b, now, dpad});
        } else {
            bool left = x < g_w * 0.5f;
            uint64_t b = left ? 0 : touch_button(x, y);
            g_touches.push_back({AMotionEvent_getPointerId(e, index), x, y, x, y, left ? kLeftStick : kNoStick,
                                 b, now, false});
        }
    } else if (kind == AMOTION_EVENT_ACTION_UP || kind == AMOTION_EVENT_ACTION_POINTER_UP ||
               kind == AMOTION_EVENT_ACTION_CANCEL) {
        int32_t id = AMotionEvent_getPointerId(e, index);
        const bool all = kind == AMOTION_EVENT_ACTION_UP || kind == AMOTION_EVENT_ACTION_CANCEL;
        g_touches.erase(std::remove_if(g_touches.begin(), g_touches.end(),
                                       [&](const Touch& t) {
                                           if (!all && t.id != id) return false;
                                           release_touch(t, now);
                                           return true;
                                       }),
                        g_touches.end());
    } else if (kind == AMOTION_EVENT_ACTION_MOVE) {
        size_t drawn = 0;
        const TouchButton* btns = touch_buttons(drawn);
        for (size_t k = 0; k < AMotionEvent_getPointerCount(e); k++) {
            int32_t id = AMotionEvent_getPointerId(e, k);
            for (auto& t : g_touches)
                if (t.id == id) {
                    t.x = AMotionEvent_getX(e, k);
                    t.y = AMotionEvent_getY(e, k);
                    if (drawn && t.stick == kNoStick) {
                        uint64_t hit = touch_hit(t.x / g_w, t.y / g_h, g_w / g_h);
                        if (t.dpad) {
                            constexpr uint64_t kDpadMask = kUp | kDown | kLeft | kRight;
                            if (hit & kDpadMask) {
                                t.button = hit & kDpadMask;
                            } else {
                                float cx = 0, cy = 0;
                                int dn = 0;
                                for (size_t bi = 0; bi < drawn; bi++)
                                    if (btns[bi].button & kDpadMask) cx += btns[bi].x, cy += btns[bi].y, dn++;
                                if (dn > 0) {
                                    cx /= dn, cy /= dn;
                                    float dx = (t.x / g_w - cx) * (g_w / g_h), dy = t.y / g_h - cy;
                                    float dist = std::hypot(dx, dy);
                                    if (dist > 0.02f && dist < 0.22f) {
                                        uint64_t dir = 0;
                                        if (dx > 0.38f * dist) dir |= kRight;
                                        else if (dx < -0.38f * dist) dir |= kLeft;
                                        if (dy > 0.38f * dist) dir |= kDown;
                                        else if (dy < -0.38f * dist) dir |= kUp;
                                        if (dir) t.button = dir;
                                    }
                                }
                            }
                        } else if (hit) {
                            t.button = hit;
                        }
                    }
                }
        }
    }
    return 1;
}

bool headless() {  // NX_HEADLESS (platform.h)
    static const bool on = std::getenv("NX_HEADLESS") != nullptr;
    return on;
}

void read_pad(PadState& p) {
    std::lock_guard lk(g_in);
    p.buttons |= g_pads[0].buttons | g_kbd_buttons.buttons;
    if (g_pads[0].lx || g_pads[0].ly) {
        p.lx = g_pads[0].lx;
        p.ly = g_pads[0].ly;
    }
    p.rx = g_pads[0].rx;
    p.ry = g_pads[0].ry;
    float radius = std::max(1.f, g_h * 0.12f);
    size_t drawn = 0;
    touch_buttons(drawn);
    const uint64_t now = monotonic_ns();
    g_latched.erase(std::remove_if(g_latched.begin(), g_latched.end(), [&](const Latch& l) { return now >= l.until_ns; }),
                    g_latched.end());
    for (const auto& l : g_latched) p.buttons |= l.button;
    for (const auto& t : g_touches) {
        float dx = (t.x - t.x0) / radius, dy = (t.y - t.y0) / radius;
        if (t.stick == kLeftStick) {
            float sx = dx * stick_sens();
            float sy = -dy * stick_sens();
            p.lx = int32_t(std::clamp(sx, -1.f, 1.f) * kStickMax);
            p.ly = int32_t(std::clamp(sy, -1.f, 1.f) * kStickMax);
            // A drawn layout's stick pushed nearly all the way may hold a button too (Wonder: Y, dash), so running
            // needs no second thumb on the right side.
            if (drawn && std::abs(dx) > 0.85f) p.buttons |= touch_stick_full_button();
        } else if (t.stick == kRightStick) {
            float sx = dx * camera_sens();
            float sy = -dy * camera_sens();
            p.rx = int32_t(std::clamp(sx, -1.f, 1.f) * kStickMax);
            p.ry = int32_t(std::clamp(sy, -1.f, 1.f) * kStickMax);
        } else {
            p.buttons |= t.button;
        }
    }
    input_script(p);  // NX_INPUT_SCRIPT (config.txt): scripted runs on the phone, as on the Mac
}

bool read_pad2(PadState& p) {
    static uint64_t next_check_ns = 0;
    std::lock_guard lk(g_in);
    if (g_pad_dev[1] < 0) return false;
    const uint64_t now = monotonic_ns();
    if (now >= next_check_ns) {
        next_check_ns = now + 2'000'000'000ull;
        if (!device_connected(g_pad_dev[1])) {
            log("hid: android pad 2 disconnected (device %d)", g_pad_dev[1]);
            g_pad_dev[1] = -1;
            g_pads[1] = PadState{};
            return false;
        }
    }
    p.buttons |= g_pads[1].buttons;
    if (g_pads[1].lx || g_pads[1].ly) {
        p.lx = g_pads[1].lx;
        p.ly = g_pads[1].ly;
    }
    p.rx = g_pads[1].rx;
    p.ry = g_pads[1].ry;
    input_script(p, 1);  // NX_INPUT_SCRIPT_P2
    return true;
}

// The drawn layout's buttons (brighter while pressed) and the stick under the thumb; nothing while a controller is
// in use or for games without a drawn layout.
size_t touch_overlay(OverlayShape* out, size_t max) {
    size_t n = 0, k = 0;
    const TouchButton* b = touch_buttons(n);
    if (!n) return 0;
    std::lock_guard lk(g_in);
    if (g_gamepad || !touch_enabled() || touch_hidden()) return 0;
    // Dimmed to a third while nobody touches the screen (cutscenes, videos), back at once on the next touch.
    const bool idle = g_touches.empty() && monotonic_ns() - g_last_touch_ns > kTouchIdleNs;
    const float shade = touch_opacity_scale() * (idle ? 0.35f : 1.0f);
    const Touch* stick = nullptr;
    const Touch* camera = nullptr;
    uint64_t held = 0;
    for (const auto& t : g_touches) {
        if (t.stick == kLeftStick) stick = &t;
        if (t.stick == kRightStick) camera = &t;
        held |= t.button;
    }
    for (const auto& l : g_latched) held |= l.button;
    for (size_t i = 0; i < n && k < max; i++)
        out[k++] = {b[i].x, b[i].y, b[i].r, std::min(1.0f, (held & b[i].button ? 0.5f : 0.2f) * shade), b[i].label};
    // A stick under the thumb: its ring where the finger went down and the knob where it is now.
    auto draw_stick = [&](const Touch& t) {
        if (k + 2 > max) return;
        float r = 0.12f, dx = (t.x - t.x0) / g_h, dy = (t.y - t.y0) / g_h;
        float len = std::hypot(dx, dy), s = len > r ? r / len : 1.f;
        out[k++] = {t.x0 / g_w, t.y0 / g_h, r, std::min(1.0f, 0.14f * shade)};
        out[k++] = {(t.x0 / g_h + dx * s) * g_h / g_w, t.y0 / g_h + dy * s, 0.055f, std::min(1.0f, 0.45f * shade)};
    };
    if (stick) {
        draw_stick(*stick);
    } else if (k < max) {  // where the stick goes
        float sx, sy, sr;
        touch_stick_rest(sx, sy, sr);
        out[k++] = {sx, sy, sr, std::min(1.0f, 0.08f * shade)};
    }
    if (camera) draw_stick(*camera);
    return k;
}

}  // namespace nx::plat
