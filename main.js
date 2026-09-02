const path = require('path');
const fs = require('fs');
const { pathToFileURL } = require('url');
const { execFile } = require('child_process');
const { app, BrowserWindow, ipcMain, safeStorage, protocol, net, session, Menu, screen, globalShortcut, desktopCapturer, systemPreferences, nativeImage } = require('electron');

const Anthropic = require('@anthropic-ai/sdk');

// ----- Serving the app over a custom "iris://" protocol instead of file:// -----
// Gaze tracking needs the webcam (getUserMedia), and Electron/Chromium
// reliably blocks getUserMedia on file:// pages — a well-known Electron
// gotcha, not a bug in our code. The fix is to serve our own HTML/JS over a
// custom protocol registered as "standard" and "secure" instead, which
// Chromium treats like a normal secure origin. This has to be registered
// before the app is ready — Electron enforces that timing.
protocol.registerSchemesAsPrivileged([
  {
    scheme: 'iris',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      stream: true, // large files (the MediaPipe WASM + face_landmarker.task model) are served as streams
    },
  },
]);

// The client is created once we actually know the API key — in dev that's
// immediate (see getDevApiKey below), but in a packaged app it may not
// exist until the user finishes the one-time setup window.
let anthropic = null;

// ----- Where the encrypted key lives in a PACKAGED app -----
// app.getPath('userData') is a per-user folder OUTSIDE the installed .app
// bundle (e.g. ~/Library/Application Support/IRIS on macOS) — nothing here
// is ever bundled into the app or visible by right-clicking the app and
// choosing "Show Package Contents".
function getSecretFilePath() {
  return path.join(app.getPath('userData'), 'iris-secret.enc');
}

// Reads the key back and decrypts it with safeStorage, which is Electron's
// built-in wrapper around the OS's own credential store (Keychain on
// macOS, DPAPI on Windows) — the key is encrypted at rest with a key WE
// never see or manage ourselves, so the file on disk isn't plain text.
function loadPackagedApiKey() {
  const secretFile = getSecretFilePath();
  if (!fs.existsSync(secretFile)) return null;
  if (!safeStorage.isEncryptionAvailable()) return null;
  try {
    const encrypted = fs.readFileSync(secretFile);
    return safeStorage.decryptString(encrypted);
  } catch (error) {
    console.error('Failed to decrypt the stored API key:', error);
    return null;
  }
}

function savePackagedApiKey(key) {
  const encrypted = safeStorage.encryptString(key);
  fs.writeFileSync(getSecretFilePath(), encrypted);
}

// Dev mode only: loads variables from the .env file in this folder (the
// same as before). This file is never bundled into a packaged build (see
// the "files" exclusions in package.json), so this path is skipped
// entirely once the app is packaged.
function getDevApiKey() {
  const result = require('dotenv').config();
  const key = process.env.ANTHROPIC_API_KEY || null;
  // Says where the key did or did not come from. Never the key itself — only
  // its length and prefix, which is enough to tell "wrong key" from "no key"
  // and from "key with a stray newline or quote".
  console.log('[claude] dotenv:', result.error
    ? `no .env loaded (${result.error.code || result.error.message})`
    : `.env loaded, ${Object.keys(result.parsed || {}).length} var(s)`);
  console.log('[claude] ANTHROPIC_API_KEY:', describeApiKey(key));
  return key;
}

// A safe description of a key, for the log. Trailing whitespace is called out
// specifically: a key pasted into .env with a trailing space or quote is one
// of the commonest causes of a 401 that looks like "the key is right there".
function describeApiKey(key) {
  if (!key) return 'MISSING';
  const trimmed = key.trim();
  const notes = [];
  if (trimmed !== key) notes.push('HAS SURROUNDING WHITESPACE');
  if (/^["']|["']$/.test(trimmed)) notes.push('HAS QUOTE CHARACTERS');
  if (!trimmed.startsWith('sk-ant-')) notes.push('DOES NOT START WITH sk-ant-');
  return `present, ${key.length} chars, starts "${key.slice(0, 11)}…"` +
    (notes.length ? ` — ${notes.join(', ')}` : '');
}

// ===== Windows =====
//
// IRIS runs as TWO windows with a strict division of labour:
//
//   main window (index.html) — owns everything. The camera, the gaze
//     pipeline, the dwell timer, the CALM/DRIFTING/FOCUSED/STUCK decision,
//     the dashboard and the debug readout. It can be hidden; it keeps
//     tracking while hidden (see setMainWindowVisible below for how).
//
//   orb window (orb.html) — owns nothing. A transparent, frameless,
//     always-on-top, click-through window containing only the orb. It parks
//     at the edge of the screen and never moves; it changes colour and
//     pulse when the main window tells it the attention state changed.
//
// Renderers never talk to each other directly — every message goes
// main window -> main process -> orb window, over the IPC relay further down.

// Module-level handles so the IPC handlers and global shortcuts below can
// reach either window without hunting through BrowserWindow.getAllWindows().
let mainWindow = null;
let orbWindow = null;
// Why the orb is currently hidden, kept apart so the two reasons cannot undo
// each other: the reader pressed O, or the main window has focus and the
// floating copy would only be in the way.
let orbHiddenByUser = false;
let orbHiddenByFocus = false;

// The last attention state the main window reported. Cached here so an orb
// window created (or re-shown) mid-session can be handed the CURRENT state
// immediately, instead of sitting on its default CALM until whenever the
// next state CHANGE happens to fire — which, if you are sitting still,
// could be a long time.
let lastOrbState = 'calm';
// The orb style chosen in the dashboard, held here for the same reason as
// lastOrbState: the orb window can be destroyed and recreated at any time, and
// it has to come back looking the way the user left it rather than reverting
// to the default baked into orb.html's markup.
let lastOrbStyle = 'classic';

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      // The preload script is the ONLY bridge between the page and Node/Electron.
      preload: path.join(__dirname, 'preload.js'),
      // These two (Electron's defaults, set explicitly here) are what make it
      // safe to expose that bridge: the page can't reach Node APIs directly,
      // and its JS world is isolated from the preload script's JS world.
      contextIsolation: true,
      nodeIntegration: false,
      // The entire gaze pipeline is requestAnimationFrame-driven — the frame
      // pump in gaze.js and the tick() loop in index.html. Chromium throttles
      // rAF and timers hard for a window that isn't the foreground one, which
      // would stall tracking exactly when it matters most: while you are
      // working in another app with only the orb on screen. Turning that
      // throttle off is what lets the main window sit in the background and
      // keep feeding the orb.
      backgroundThrottling: false,
    },
  });

  mainWindow.loadURL('iris://app/index.html');

  // The main window is an ordinary app window: it belongs in the Dock, it
  // minimises and restores like anything else, and it appears in the app
  // switcher. Nothing about the orb's overlay behaviour is applied to it.
  //
  // The one thing that needs guarding is the "still tracking" hide, which
  // leaves this window at opacity 0 and click-through (see
  // setMainWindowVisible). If it is put away that way and then brought back
  // by any route that does NOT go through setMainWindowVisible — clicking the
  // Dock icon, the app switcher, restoring from a minimise — the window is
  // shown but completely invisible, which looks exactly like it has vanished.
  // Re-asserting normal chrome on every show/restore closes off all of those
  // routes at once.
  const restoreMainWindowChrome = () => {
    if (mainWindow.isDestroyed()) return;
    mainWindow.setOpacity(1);
    mainWindow.setIgnoreMouseEvents(false);
  };
  mainWindow.on('show', restoreMainWindowChrome);
  mainWindow.on('restore', restoreMainWindowChrome);

  // ----- The overlay yields to the app -----
  // The orb exists to be visible while you are reading in ANOTHER app. While
  // the main window is focused you are looking at IRIS itself, which draws its
  // own in-page orb — so the floating copy has nothing to add and everything
  // to obscure. Hiding it on focus is what guarantees that switching between
  // Orb and Dashboard happens inside one window, with no second window over
  // the top of it. `orbHiddenByUser` is respected either way, so the O
  // shortcut still wins.
  mainWindow.on('focus', () => {
    if (orbWindow && !orbWindow.isDestroyed() && orbWindow.isVisible()) {
      orbHiddenByFocus = true;
      setOrbInteractive(false);
      orbWindow.hide();
    }
  });

  mainWindow.on('blur', () => {
    if (orbHiddenByFocus && !orbHiddenByUser) {
      orbHiddenByFocus = false;
      setOrbVisible(true);
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
    // The orb is a readout of this window's gaze loop. With the main window
    // actually CLOSED (not merely hidden) there is nothing left to drive it,
    // so a lingering orb would just be a frozen light stuck on top of
    // everything with no way to change or dismiss it.
    if (orbWindow && !orbWindow.isDestroyed()) orbWindow.close();
  });

  // The orb comes up with the app. Both windows are visible on launch: the
  // orb floating at the screen edge, the main window behind it.
  createOrbWindow();
}

// ----- The orb window -----
// Sized to hold two things: the orb, and the help card that grows out of it
// when a STUCK moment fires.
//
// The orb alone would need roughly 300px square — its core is 56px, but the
// STUCK glow reaches ~136px past that box (a 100px blur with 36px of spread)
// and the ripple ring scales to 2.8x, and anything outside the window is
// clipped into a hard straight edge. The card then extends to the right of
// the orb, which is what the extra width and height are for.
//
// The window is this size ALWAYS, card or no card. Growing it on demand would
// mean resizing a transparent always-on-top window mid-animation, and a window
// that is entirely click-through and entirely transparent costs nothing to
// leave large — there is no visible box, and no clicks are intercepted.
const ORB_WINDOW_WIDTH = 520;
const ORB_WINDOW_HEIGHT = 380;

// Distance from the orb window's top-left corner to the CENTRE of the orb
// core drawn inside it. This MUST match the `top`/`left` on #orb-core in
// orb.html: main.js parks the window, orb.html places the orb within it, and
// where the orb actually lands on screen is the sum of the two.
//
// It also decides how much room the glow has on the top and left sides. The
// glow reaches roughly 120px from the core at its widest (the STUCK state),
// so at 80px it is clipped where it is already faint, while still leaving the
// orb sitting comfortably in the corner rather than jammed against it.
const ORB_CORE_INSET = 80;

function createOrbWindow() {
  if (orbWindow && !orbWindow.isDestroyed()) return orbWindow;

  orbWindow = new BrowserWindow({
    width: ORB_WINDOW_WIDTH,
    height: ORB_WINDOW_HEIGHT,
    // No frame, no title bar, no drop shadow, and a fully transparent
    // backdrop, so the only thing that renders anywhere in this 300px square
    // is the orb's own glow. Without `hasShadow: false` the OS draws a
    // rectangular shadow around the invisible window and the illusion dies.
    frame: false,
    transparent: true,
    hasShadow: false,
    backgroundColor: '#00000000',
    // It's an indicator, not a window: it can't be resized, dragged,
    // minimised, maximised or fullscreened, and it never appears in the
    // Dock/taskbar or the app switcher.
    resizable: false,
    movable: false,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    // Windows/Linux only. There is no per-window taskbar on macOS, so
    // Electron maps skipTaskbar onto the APP's Dock icon — which is app-level
    // state, not this window's. Setting it here hid the whole of IRIS from
    // the Dock and the app switcher, taking the main window with it: minimise
    // the dashboard and there was nowhere left to click to get it back. And
    // it does not undo — calling setSkipTaskbar(false) afterwards leaves the
    // icon gone. The orb has no business in a taskbar either way, but on
    // macOS a frameless, non-focusable window already stays out of the
    // window menu and app switcher on its own, so nothing is needed here.
    skipTaskbar: process.platform !== 'darwin',
    // Never takes keyboard focus. Without this, the orb appearing would pull
    // focus away from whatever you're actually reading or typing in.
    focusable: false,
    // Deliberately not `show: true` — showInactive() below puts it on screen
    // WITHOUT activating it, for the same reason.
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      // The orb's pulse is a CSS animation, which is also rAF-paced. Same
      // reasoning as the main window: it must keep animating while some other
      // app is in the foreground, which is essentially always.
      backgroundThrottling: false,
    },
  });

  // Click-through. Every mouse event passes straight to whatever is
  // underneath, so this 300px square floating over your screen never eats a
  // click, a hover, a drag or a text selection. `forward: true` still lets the
  // page observe move events (harmless here, and it keeps CSS :hover working
  // if this ever grows a hover affordance).
  orbWindow.setIgnoreMouseEvents(true, { forward: true });

  // Applied here, right after the window exists, and again on every show
  // (below) — see applyOrbOverlayBehaviour() for why once is not enough.
  applyOrbOverlayBehaviour();

  orbWindow.loadURL('iris://app/orb.html');

  // Re-assert the overlay behaviour every time the window is shown, not just
  // when it is built. hide() orders the window out on macOS, and it does not
  // reliably come back with its collection behaviour intact — so the orb
  // could survive a full-screen app on first launch and then vanish behind
  // one after a single hide/show cycle. 'show' fires for showInactive() too,
  // so this covers the first appearance and every later un-hide, including
  // the Cmd+Shift+O toggle and setOrbVisible().
  orbWindow.on('show', applyOrbOverlayBehaviour);

  orbWindow.once('ready-to-show', () => {
    positionOrbWindow();
    // showInactive(), not show(): puts it on screen without stealing focus.
    orbWindow.showInactive();
    // Come up in the state we're ACTUALLY in, not the CALM default baked
    // into orb.html's markup.
    orbWindow.webContents.send('orb-state', lastOrbState);
    orbWindow.webContents.send('orb-style', lastOrbStyle);
  });

  orbWindow.on('closed', () => {
    orbWindow = null;
  });

  return orbWindow;
}

// ----- Surviving another app going full screen -----
// These are the two settings that decide whether the orb stays on screen when
// something else takes over the display. They live in one function because
// they have to be REAPPLIED rather than set once: they are properties of the
// native window, and macOS does not reliably preserve them across a window
// being ordered out and back in. Setting them only at creation time is how
// the orb ends up vanishing behind a full-screen app.
//
// The ORDER matters. setVisibleOnAllWorkspaces rewrites the window's
// collection behaviour — the flag that lets it join another app's full-screen
// Space at all — and on macOS that can reset the window's level as a side
// effect. Setting the level second means the level is the one that survives.
function applyOrbOverlayBehaviour() {
  if (!orbWindow || orbWindow.isDestroyed()) return;

  // Follows you across Spaces and floats over full-screen apps. This is the
  // whole point of an ambient indicator: it shouldn't vanish because you
  // switched desktop or went full screen in your reader.
  orbWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  // 'screen-saver' is the highest of the ordinary always-on-top levels: it
  // keeps the orb above normal windows AND above other apps' always-on-top
  // windows, which a bare setAlwaysOnTop(true) does not.
  orbWindow.setAlwaysOnTop(true, 'screen-saver');
}

// Parks the orb window in the TOP-LEFT corner of the screen. This runs on
// creation, on every show, and whenever the display layout changes — it is
// the ONLY thing that ever positions the orb. The orb does not follow your
// gaze; where you look changes its colour and pulse, never its position.
function positionOrbWindow() {
  if (!orbWindow || orbWindow.isDestroyed()) return;

  // Park on whichever display the main window is on, so the orb doesn't end
  // up on a monitor you aren't using. workArea (not bounds) already excludes
  // the menu bar, Dock and taskbar, so "the edge" means the edge of the
  // usable screen rather than a spot half-under a system bar.
  const display = (mainWindow && !mainWindow.isDestroyed())
    ? screen.getDisplayMatching(mainWindow.getBounds())
    : screen.getPrimaryDisplay();
  const area = display.workArea;

  // Flush into the corner of the work area. The window is much larger than
  // the orb so the glow has room to spread without being clipped, and the orb
  // is offset INSIDE it (ORB_CORE_INSET, applied in orb.html) rather than
  // centred — which is what puts the visible orb in the corner instead of
  // 150px inland.
  //
  // Offsetting inside the window is necessary rather than merely tidy: the
  // obvious alternative — centring the orb and letting the window hang off
  // the top and left — does not work, because macOS clamps a window's origin
  // to the work area. Asking for a negative x, or a y above the menu bar,
  // silently snaps the window back to the corner and drags the orb 150px
  // inland with it. Positioning at the right or bottom edge CAN overhang;
  // top and left cannot.
  // Flush into the TOP-LEFT of the work area, which is where the orb itself
  // then lands: orb.html insets the core ORB_CORE_INSET from the window's own
  // top-left corner, so window origin plus that inset IS the visible orb.
  //
  // It was moved to the bottom edge for a while to stop it sitting over the
  // main window's nav bar. That put the window's top-left — and therefore the
  // orb — ORB_WINDOW_HEIGHT above the bottom of the screen, which reads as
  // floating in the middle of the left-hand side rather than parked in a
  // corner. The nav overlap is handled properly now by hiding the overlay
  // whenever the main window has focus (see the 'focus'/'blur' handlers in
  // createMainWindow), so the corner is free again.
  orbWindow.setBounds({
    x: area.x,
    y: area.y,
    width: ORB_WINDOW_WIDTH,
    height: ORB_WINDOW_HEIGHT,
  });
}

// ----- Showing and hiding the orb -----
// Hiding the orb is a plain hide(): it holds no state of its own, so there is
// nothing to lose, and lastOrbState above means it comes back correct.
function isOrbVisible() {
  return !!(orbWindow && !orbWindow.isDestroyed() && orbWindow.isVisible());
}

function setOrbVisible(visible) {
  if (!visible) {
    if (orbWindow && !orbWindow.isDestroyed()) {
      // Never leave a hidden window still claiming mouse events.
      setOrbInteractive(false);
      orbWindow.hide();
    }
    return false;
  }

  if (!orbWindow || orbWindow.isDestroyed()) {
    createOrbWindow(); // its ready-to-show handler positions, shows and syncs it
    return true;
  }

  positionOrbWindow(); // in case displays changed while it was hidden
  orbWindow.showInactive();
  orbWindow.webContents.send('orb-state', lastOrbState);
  orbWindow.webContents.send('orb-style', lastOrbStyle);
  return true;
}

// ----- Showing and hiding the MAIN window -----
// win.hide() would be the obvious implementation, and it is wrong here. A
// hidden window is not composited: Chromium stops producing frames for it, so
// requestAnimationFrame stops firing — and the entire gaze pipeline is
// rAF-driven (gaze.js's frame pump, index.html's tick()). Hiding the main
// window that way would silently freeze tracking, and the orb would sit there
// glowing a stale colour. backgroundThrottling:false covers a merely
// BACKGROUNDED window; it cannot help a window that has been ordered off the
// screen entirely.
//
// So the default "hidden" here means invisible and inert rather than gone:
// opacity 0 (still composited, so rAF keeps running and gaze keeps tracking)
// plus click-through (so an invisible full-size rectangle never swallows a
// click). Pass { full: true } for a real hide() — out of the Dock and the app
// switcher — accepting that tracking pauses until it comes back.
// The "I've lost the window" escape hatch. Unlike the Cmd/Ctrl+Shift+D
// toggle, this only ever SHOWS — a toggle is no use when you cannot tell
// whether the window is hidden or merely buried, since half the time it would
// hide the thing you are trying to find. It digs the window out of every state
// it can get into: minimised, stealth-hidden at opacity 0, behind everything
// else, or with the Dock icon missing from an older build.
function recoverMainWindow() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    createMainWindow();
    return;
  }

  if (process.platform === 'darwin' && app.dock) app.dock.show();
  if (mainWindow.isMinimized()) mainWindow.restore();
  setMainWindowVisible(true);
  mainWindow.moveTop();

  // On macOS a window cannot come to the front while its app is in the
  // background, so the app has to be raised too — and `steal` is what makes
  // that work when the keystroke arrived while another app was active, which
  // is the whole point of a global shortcut.
  if (process.platform === 'darwin') app.focus({ steal: true });
}

function isMainWindowVisible() {
  if (!mainWindow || mainWindow.isDestroyed()) return false;
  return mainWindow.isVisible() && mainWindow.getOpacity() > 0;
}

function setMainWindowVisible(visible, options = {}) {
  if (!mainWindow || mainWindow.isDestroyed()) return false;

  if (visible) {
    mainWindow.setIgnoreMouseEvents(false);
    mainWindow.setOpacity(1);
    if (!mainWindow.isVisible()) mainWindow.show();
    mainWindow.focus();
    return true;
  }

  if (options.full) {
    mainWindow.hide();
  } else {
    mainWindow.setOpacity(0);
    mainWindow.setIgnoreMouseEvents(true, { forward: true });
    // Drop it out of the foreground so an invisible window isn't sitting in
    // front of what you're reading in the stacking order.
    mainWindow.blur();
  }
  return false;
}

// The one-time setup window, shown only in a packaged build with no key
// stored yet. It's a separate small window, not part of the main app.
function createSetupWindow() {
  const win = new BrowserWindow({
    width: 480,
    height: 380,
    resizable: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.loadURL('iris://app/setup.html');
}

// ----- Application menu -----
// Electron's default menu binds Cmd/Ctrl+R to "Reload" and Cmd/Ctrl+Shift+R
// to "Force Reload". For a normal browser tab that's harmless; for IRIS a
// reload silently wipes all in-memory state (the camera stream, tracking
// session, calibration) with zero warning — actively harmful here, and a
// very plausible explanation for calibration appearing to "restart itself"
// on its own: a Cmd+R meant as a natural "recalibrate" shortcut would
// instead reload the whole page and re-run the boot sequence. We build our
// own menu without those two items so Cmd+R is free to bind to our own
// recalibrate shortcut instead (see index.html's keydown listener) without
// Electron's default menu intercepting it first.
function buildAppMenu() {
  const isMac = process.platform === 'darwin';
  const template = [
    ...(isMac ? [{ role: 'appMenu' }] : []),
    { role: 'editMenu' }, // keeps cut/copy/paste working in setup.html's key input
    {
      label: 'View',
      submenu: [
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    { role: 'windowMenu' },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

app.whenReady().then(() => {
  buildAppMenu();

  // ----- The Dock icon -----
  // Two separate things have to be true for it to appear, and only the second
  // was being done here.
  //
  //   1. There has to BE an icon. A packaged build gets one from the app
  //      bundle, but run straight from source (`npm start`) there is no bundle
  //      to read, and Electron falls back to an icon that macOS will not
  //      always draw — so it is set explicitly from build/icon.png, the same
  //      file electron-builder uses for the packaged app.
  //   2. The Dock has to be showing it. That is app-level state an earlier
  //      build could have cleared (see skipTaskbar in createOrbWindow), and a
  //      cleared Dock icon does not come back on its own.
  //
  // Neither step is allowed to stop the app booting: a missing or unreadable
  // icon file is a cosmetic problem, and it says so rather than throwing.
  if (process.platform === 'darwin' && app.dock) {
    const dockIconPath = path.join(__dirname, 'build', 'icon.png');
    try {
      const dockIcon = nativeImage.createFromPath(dockIconPath);
      if (dockIcon.isEmpty()) {
        console.warn('[dock] no usable icon at', dockIconPath, '- keeping the default.');
      } else {
        app.dock.setIcon(dockIcon);
      }
    } catch (err) {
      console.warn('[dock] could not set the icon:', err);
    }
    app.dock.show();
  }

  // Keep the orb parked at the edge of the RIGHT screen when the display
  // layout changes underneath it — plugging in a monitor, unplugging one, or
  // changing resolution/scaling would otherwise strand it in mid-air or
  // off-screen entirely. The `screen` module can only be used after ready.
  screen.on('display-metrics-changed', positionOrbWindow);
  screen.on('display-added', positionOrbWindow);
  screen.on('display-removed', positionOrbWindow);

  registerGlobalShortcuts();

  // Serves our own project files over iris://app/<path>, mapped straight
  // onto this folder on disk — e.g. iris://app/index.html is this folder's
  // index.html, iris://app/node_modules/@mediapipe/tasks-vision/... is the
  // real file under node_modules. Registered once, used by every window.
  protocol.handle('iris', (request) => {
    const url = new URL(request.url);
    const filePath = path.join(__dirname, decodeURIComponent(url.pathname));
    return net.fetch(pathToFileURL(filePath).toString());
  });

  // Camera access (for gaze tracking) triggers a permission request the
  // first time it's used. This is our own app's content asking, not some
  // third-party page, so we auto-approve camera/mic requests and deny
  // everything else rather than leaving Electron's default behavior implicit.
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    callback(permission === 'media');
  });

  if (!app.isPackaged) {
    // Dev mode: unchanged from before — always open the app, key or not.
    // A missing/invalid key still surfaces later as an error when you
    // actually ask for help, rather than blocking startup.
    anthropic = new Anthropic({ apiKey: getDevApiKey() || undefined });
    console.log(`[claude] client created (dev), model ${SCREEN_MODEL}`);
    createMainWindow();
    return;
  }

  // Packaged mode: only open the real app once we have a key. If none is
  // stored yet, show the setup window instead — it saves the key via the
  // 'save-api-key' handler below, which is what actually opens the main window.
  const key = loadPackagedApiKey();
  console.log('[claude] stored key:', describeApiKey(key));
  if (key) {
    anthropic = new Anthropic({ apiKey: key });
    console.log(`[claude] client created (packaged), model ${SCREEN_MODEL}`);
    createMainWindow();
  } else {
    createSetupWindow();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Lets the renderer ask "is this window currently maximised (or full
// screen)?" — used to block calibration in a non-maximised window, since
// calibration is only valid for the exact window size it ran at, and a
// maximised window is the one size the user can reliably return to.
// Only the main process can know this; the renderer's own
// window.innerWidth-vs-screen heuristics are unreliable across menu bars,
// docks, and displays.
ipcMain.handle('is-window-maximized', (event) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (!win) return false;
  return win.isMaximized() || win.isFullScreen();
});

// Handles the setup window's submitted key: saves it (encrypted) for next
// time, creates the real Anthropic client, then swaps the setup window for
// the main app window.
ipcMain.handle('save-api-key', async (event, rawKey) => {
  const key = (rawKey || '').trim();
  if (!key) {
    return { ok: false, error: 'Please paste a valid API key.' };
  }

  try {
    savePackagedApiKey(key);
    anthropic = new Anthropic({ apiKey: key });

    createMainWindow();
    BrowserWindow.fromWebContents(event.sender)?.close();

    return { ok: true };
  } catch (error) {
    return { ok: false, error: error.message || 'Failed to save the key.' };
  }
});


// ===== The help card, in the orb window =====
//
// A STUCK moment used to open a panel inside the MAIN window, which is exactly
// where it could not be seen: the main window is usually hidden behind whatever
// is being read. The card now lives in the orb window instead, so it appears
// over the front-most app.
//
// The division of labour is unchanged in spirit — the main window is still the
// brain and the orb window is still a pure view:
//
//   main window  decides when STUCK fires, owns the struggle-event log, runs
//                the Claude request, and owns the gaze-away dismissal timer
//                (it is the only window with gaze data).
//   orb window   renders the card, animates it out of the orb, reports where
//                it ended up on screen, and emits button clicks upward.
//
// Neither renderer talks to the other; every message is relayed through here.

// ----- Click-through, but only sometimes -----
// The orb window is click-through so it never eats a click meant for the app
// underneath. That has to stop being true for the card's own rectangle, or its
// buttons could not be pressed.
//
// The switch is driven by the cursor: orb.html watches mousemove (which still
// arrives while click-through, because of `forward: true`) and calls this as
// the pointer crosses into and out of the card. So the window is only ever
// mouse-opaque while the pointer is actually over the card — everywhere else,
// and at all times while no card is showing, clicks pass straight through.
function setOrbInteractive(interactive) {
  if (!orbWindow || orbWindow.isDestroyed()) return;
  orbWindow.setIgnoreMouseEvents(!interactive, { forward: true });
}

// main window -> orb window. One channel carrying {type, ...}: 'show',
// 'update' (streaming text and mode changes), 'fade', 'hide'.
ipcMain.on('orb-card', (event, message) => {
  // A card that is going away must never leave the window claiming clicks —
  // the pointer may already be somewhere else, in which case no further
  // mousemove would arrive to turn it back off.
  if (message.type === 'hide') setOrbInteractive(false);

  if (orbWindow && !orbWindow.isDestroyed()) {
    orbWindow.webContents.send('orb-card', message);
  }
});

// The renderer owns the stuck moment's lifetime, so it says when the keyboard
// fallbacks should be live. That is deliberately NOT the card's lifetime: the
// offer stage has no card, and accept has to be reachable then.
ipcMain.on('help-shortcuts', (event, active) => setHelpCardShortcuts(!!active));

// The orb's own rectangle, converted into the MAIN window's viewport
// coordinates — the space gaze predictions live in. Used to decide whether you
// are looking at the orb during the offer stage, when there is no card to look
// at. Same maximised-window caveat as the card rect above.
ipcMain.handle('get-orb-rect', () => {
  if (!orbWindow || orbWindow.isDestroyed()) return null;
  if (!mainWindow || mainWindow.isDestroyed()) return null;

  const orbBounds = orbWindow.getContentBounds();
  const mainContent = mainWindow.getContentBounds();
  const centreX = orbBounds.x + ORB_CORE_INSET;
  const centreY = orbBounds.y + ORB_CORE_INSET;

  // Generously larger than the 56px core: the glow reads as part of the orb,
  // and gaze is at its least accurate in the very corner of the screen, which
  // is exactly where the orb lives.
  const radius = 90;
  return {
    left: centreX - radius - mainContent.x,
    top: centreY - radius - mainContent.y,
    right: centreX + radius - mainContent.x,
    bottom: centreY + radius - mainContent.y,
  };
});

// ===== Low-resolution screen fingerprint =====
// Used to notice that the content under the reader changed — a scroll, or a
// different app. 64x40 greyscale is 2560 numbers for the whole display: far
// too coarse to read anything off (a line of text is well under one pixel
// tall), and enough to answer the only question asked of it, which is whether
// this is a different picture. Nothing is stored.
const SCREEN_FINGERPRINT_WIDTH = 64;
const SCREEN_FINGERPRINT_HEIGHT = 40;

ipcMain.handle('screen-fingerprint', async () => {
  try {
    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: { width: SCREEN_FINGERPRINT_WIDTH, height: SCREEN_FINGERPRINT_HEIGHT },
    });
    if (!sources.length || sources[0].thumbnail.isEmpty()) return null;
    const bitmap = sources[0].thumbnail.toBitmap(); // BGRA
    const fingerprint = new Array(Math.floor(bitmap.length / 4));
    for (let i = 0, p = 0; i + 2 < bitmap.length; i += 4, p += 1) {
      fingerprint[p] = (bitmap[i] + bitmap[i + 1] + bitmap[i + 2]) / 3;
    }
    return fingerprint;
  } catch (error) {
    // A background heuristic degrades to "no signal" rather than surfacing.
    console.warn('[content] screen fingerprint failed:', error.message);
    return null;
  }
});

// ===== Calibration persistence =====
// Four coefficients and the window size they were trained at. Stored in
// userData rather than beside the app, so it survives reinstalling and is
// per-user; JSON rather than anything cleverer, because it is four numbers and
// being able to read or delete it by hand is worth more than compactness.
function calibrationFilePath() {
  return path.join(app.getPath('userData'), 'calibration.json');
}

ipcMain.handle('save-calibration', (event, payload) => {
  try {
    fs.writeFileSync(calibrationFilePath(), JSON.stringify(payload, null, 2));
    console.log(`[calibration] saved to ${calibrationFilePath()}`);
    return { ok: true, path: calibrationFilePath() };
  } catch (error) {
    console.error('[calibration] could not save:', error.message);
    return { ok: false, error: error.message };
  }
});

ipcMain.handle('load-calibration', () => {
  const file = calibrationFilePath();
  if (!fs.existsSync(file)) return { ok: false, missing: true, path: file };
  try {
    return { ok: true, path: file, data: JSON.parse(fs.readFileSync(file, 'utf8')) };
  } catch (error) {
    // A corrupt file should not block startup — report it and carry on
    // uncalibrated, which is the state the app already knows how to be in.
    console.error('[calibration] stored file is unreadable:', error.message);
    return { ok: false, error: error.message, path: file };
  }
});

ipcMain.handle('clear-calibration', () => {
  const file = calibrationFilePath();
  try {
    if (fs.existsSync(file)) fs.unlinkSync(file);
    console.log('[calibration] stored calibration deleted.');
    return { ok: true, path: file };
  } catch (error) {
    console.error('[calibration] could not delete:', error.message);
    return { ok: false, error: error.message };
  }
});

// ===== Zone metric logging =====
//
// The debug panel is in the main window, which sits behind whatever is being
// read — so the numbers are invisible exactly when they matter. This writes
// them to a CSV instead, twice a second, to be read afterwards.
//
// The main process owns the file because only it can touch the disk; the
// renderer owns the numbers and sends finished rows. Nothing is sampled here.
let zoneLogStream = null;
let zoneLogPath = null;
let zoneLogStartedAt = 0;
let zoneLogRows = 0;

const ZONE_LOG_HEADER = 'iso_time,ms_elapsed,zone,progress,progress_median,progress_ratio,idle,row_cycles,sustain_ms,content_diff,revisits,concentration,marker\n';

function zoneLogDirectory() {
  return path.join(app.getPath('userData'), 'logs');
}

ipcMain.handle('zone-log-start', () => {
  if (zoneLogStream) return { ok: true, path: zoneLogPath, alreadyRunning: true };
  try {
    const directory = zoneLogDirectory();
    fs.mkdirSync(directory, { recursive: true });
    // One file per session, named by when it started, so successive runs never
    // overwrite each other and the filename alone says which sitting it was.
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    zoneLogPath = path.join(directory, `zones-${stamp}.csv`);
    zoneLogStream = fs.createWriteStream(zoneLogPath, { flags: 'a' });
    zoneLogStream.write(ZONE_LOG_HEADER);
    zoneLogStartedAt = Date.now();
    zoneLogRows = 0;
    console.log(`[zone-log] recording to: ${zoneLogPath}`);
    console.log(`[zone-log]   open it with:  open "${zoneLogPath}"`);
    console.log(`[zone-log]   reveal it with: open -R "${zoneLogPath}"`);
    return { ok: true, path: zoneLogPath };
  } catch (error) {
    console.error('[zone-log] could not start:', error.message);
    zoneLogStream = null;
    return { ok: false, error: error.message };
  }
});

ipcMain.on('zone-log-row', (event, row) => {
  if (!zoneLogStream) return;
  zoneLogRows += 1;
  // Written as sent. The renderer decides what a row means; this only records.
  zoneLogStream.write(
    `${row.isoTime},${row.msElapsed},${row.zone},${row.progress},` +
    `${row.progressMedian},${row.progressRatio},` +
    `${row.idle},${row.rowCycles},${row.sustainMs},${row.contentDiff},` +
    `${row.revisits},${row.concentration},${row.marker || ''}\n`,
  );
});

ipcMain.handle('zone-log-stop', () => {
  if (!zoneLogStream) return { ok: true, path: null, rows: 0 };
  const finishedPath = zoneLogPath;
  const rows = zoneLogRows;
  zoneLogStream.end();
  zoneLogStream = null;
  console.log(`[zone-log] stopped after ${rows} rows — ${finishedPath}`);
  return { ok: true, path: finishedPath, rows, durationMs: Date.now() - zoneLogStartedAt };
});

// A half-written CSV is still readable, but only if the buffer reaches disk.
app.on('will-quit', () => {
  if (zoneLogStream) {
    zoneLogStream.end();
    zoneLogStream = null;
  }
});

// The orb window's optional readout — relayed straight through, like the card.
ipcMain.on('orb-readout', (event, payload) => {
  if (orbWindow && !orbWindow.isDestroyed()) {
    orbWindow.webContents.send('orb-readout', payload);
  }
});

// ----- Keyboard fallbacks for the card, while a stuck moment is live -----
// The card's own keys (Y / N / Enter / Escape, handled in index.html) only
// reach it while IRIS has keyboard focus — and the whole point of the card is
// that it appears over something else you are reading, which means IRIS
// usually does not. These global shortcuts cover that case.
//
// They are registered only for as long as a card is actually on screen. A
// global shortcut is claimed system-wide, so holding Cmd/Ctrl+Shift+Y and
// Cmd/Ctrl+Shift+N permanently would take them away from every other app for
// the sake of a card that is visible a few seconds at a time.
const HELP_CARD_SHORTCUTS = [
  ['CommandOrControl+Shift+Y', 'yes'],
  ['CommandOrControl+Shift+N', 'no'],
];
let helpCardShortcutsActive = false;

function setHelpCardShortcuts(active) {
  if (active === helpCardShortcutsActive) return;
  helpCardShortcutsActive = active;

  for (const [accelerator, action] of HELP_CARD_SHORTCUTS) {
    if (!active) {
      globalShortcut.unregister(accelerator);
      continue;
    }
    // Reuses the ordinary button path, so a shortcut, a click and a gesture
    // all arrive at the main window as the same action.
    const registered = globalShortcut.register(accelerator, () => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('orb-card-action', action);
      }
    });
    if (!registered) {
      console.warn(`[help-card] Could not register ${accelerator} — another app may already be using it.`);
    }
  }
}

// orb window -> main window: a button was pressed ('yes' | 'no' | 'close').
ipcMain.on('orb-card-action', (event, action) => {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('orb-card-action', action);
  }
});

// orb window -> main process -> main window: where the card actually is.
//
// The main window needs this to answer "are they looking at the card?", and it
// can only compare against its own viewport coordinates, because that is the
// space gaze predictions live in. So the rect makes two hops: from the orb
// window's client coordinates, out to screen coordinates, and back into the
// main window's viewport.
//
// This is only meaningful while the main window is maximised — which is
// already required for calibration, since a gaze mapping is only valid for the
// window size it was trained at. Maximised, its viewport covers the work area,
// so the card's screen position lands inside it. `null` means "no card".
ipcMain.on('orb-card-rect', (event, rect) => {
  if (!mainWindow || mainWindow.isDestroyed()) return;

  if (!rect) {
    mainWindow.webContents.send('orb-card-rect', null);
    return;
  }
  if (!orbWindow || orbWindow.isDestroyed()) return;

  const orbBounds = orbWindow.getContentBounds();
  const mainContent = mainWindow.getContentBounds();
  const left = orbBounds.x + rect.x - mainContent.x;
  const top = orbBounds.y + rect.y - mainContent.y;

  mainWindow.webContents.send('orb-card-rect', {
    left,
    top,
    right: left + rect.w,
    bottom: top + rect.h,
  });
});

// orb window -> main process. Not forwarded anywhere: this is the pointer
// crossing the card's edge, and only the main process can act on it.
ipcMain.on('orb-card-interactive', (event, interactive) => {
  setOrbInteractive(Boolean(interactive));
});

// ===== Orb window IPC =====
//
// The main window owns the attention state; the orb window displays it. The
// two never speak directly — the main process is the relay, which is the only
// way two renderers can communicate under contextIsolation.

// Sent by index.html's tick() whenever the state CHANGES (not every frame),
// from the exact same place that already swaps the main window's own orb
// class. Nothing in the gaze pipeline is altered by this: it is one extra
// send() alongside a DOM write that was already happening.
ipcMain.on('orb-state', (event, state) => {
  lastOrbState = state;
  if (orbWindow && !orbWindow.isDestroyed()) {
    orbWindow.webContents.send('orb-state', state);
  }
});

// Sent by the dashboard's Appearance panel when the orb style is changed, and
// once on load so the orb window agrees with the main window from the start.
// Relayed exactly like orb-state, and remembered for the same reason.
ipcMain.on('orb-style', (event, style) => {
  lastOrbStyle = style;
  if (orbWindow && !orbWindow.isDestroyed()) {
    orbWindow.webContents.send('orb-style', style);
  }
});

// ----- Show/hide, callable from the renderer -----
// Each returns the resulting visibility, so the caller can update its own UI
// without a second round trip.
ipcMain.handle('set-orb-visible', (event, visible) => {
  orbHiddenByUser = !visible;
  return setOrbVisible(visible);
});
ipcMain.handle('toggle-orb-visible', () => {
  const next = !isOrbVisible();
  orbHiddenByUser = !next;
  if (next) orbHiddenByFocus = false;
  return setOrbVisible(next);
});
ipcMain.handle('is-orb-visible', () => isOrbVisible());

ipcMain.handle('set-main-visible', (event, visible, options) => setMainWindowVisible(visible, options || {}));
ipcMain.handle('toggle-main-visible', () => setMainWindowVisible(!isMainWindowVisible()));
ipcMain.handle('is-main-visible', () => isMainWindowVisible());

// ----- Global shortcuts -----
// These are GLOBAL on purpose. Once the main window is hidden there is no
// IRIS window left that can receive a keypress — the orb refuses focus by
// design — so a window-scoped shortcut would leave you with no way to bring
// the dashboard back. Registering them system-wide is what makes hiding the
// main window a reversible action rather than a trap.
//
//   Cmd/Ctrl+Shift+O — show/hide the floating orb
//   Cmd/Ctrl+Shift+D — show/hide the main window (dashboard + debug view)
//   Cmd/Ctrl+Shift+I — bring the main window BACK, whatever state it is in
//   Cmd/Ctrl+Shift+J — show/hide the readout on the floating orb
//   Cmd/Ctrl+Shift+K — mark "I feel stuck" into the CSV
//   Cmd/Ctrl+Shift+L — start/stop CSV logging
//
// The third one is not a toggle on purpose: it is the recovery shortcut, for
// when you cannot tell whether the dashboard is hidden or just buried.
function registerGlobalShortcuts() {
  // The last three are global rather than window-scoped on purpose: they are
  // for use WHILE READING something else, when the main window has no keyboard
  // focus at all. J, K and L sit next to each other so they can be found
  // without looking away from the page.
  const tellRenderer = (channel) => () => {
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send(channel);
  };

  const shortcuts = [
    ['CommandOrControl+Shift+O', () => setOrbVisible(!isOrbVisible())],
    ['CommandOrControl+Shift+D', () => setMainWindowVisible(!isMainWindowVisible())],
    ['CommandOrControl+Shift+I', () => recoverMainWindow()],
    ['CommandOrControl+Shift+J', tellRenderer('zone-readout-toggle')],
    ['CommandOrControl+Shift+K', tellRenderer('zone-log-mark')],
    ['CommandOrControl+Shift+L', tellRenderer('zone-log-toggle')],
  ];

  for (const [accelerator, handler] of shortcuts) {
    // register() returns false if another app already owns the combination.
    // That is worth saying out loud rather than leaving the user pressing a
    // key that silently does nothing.
    if (!globalShortcut.register(accelerator, handler)) {
      console.warn(`[windows] Could not register the global shortcut ${accelerator} — another app may already be using it.`);
    }
  }
}

// Electron requires global shortcuts to be released explicitly; they are not
// cleaned up automatically on quit.
app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});

// macOS: clicking the Dock icon brings the main window back, including when
// it is "hidden" via opacity rather than hide(), or minimised — otherwise the
// click appears to do nothing at all.
app.on('activate', () => {
  recoverMainWindow();
});

// Tells Claude how to answer, once, instead of repeating instructions in
// every request. Kept short since the model itself is asked to be short.
const SYSTEM_PROMPT =
  'The user is stuck while reading or working through something. Answer in ' +
  '2-3 short, plain sentences they can act on immediately. No preamble, no lists.';

// ----- The one place a Messages request is sent -----
// Every call to the API goes through here, so there is exactly one object that
// can become a request body and exactly one place to inspect it.
//
// `signal` is the specific hazard. It is not a Messages parameter: it belongs
// to the HTTP request, and the SDK takes it in its SECOND argument. Put it in
// the first and it is serialised into the body, where the API rejects it as
// an unknown field — "signal: Extra inputs are not permitted", a 400 raised
// before anything else in the request is even read. It is easy to reintroduce
// by hand or through an object spread, so rather than trusting that nobody
// will, this strips it and says so.
function streamMessage(label, params, options = {}) {
  // Checked with `in`, not by truthiness: an AbortSignal is an object, and
  // JSON.stringify turns it into {} — so a body dump alone would show an
  // empty-looking key and give no hint that anything was wrong.
  let safeParams = params;
  let safeOptions = options;
  if ('signal' in params) {
    const { signal, ...rest } = params;
    safeParams = rest;
    safeOptions = { ...options, signal: options.signal || signal };
    console.error(`[claude] ${label}: "signal" was in the request BODY — this is the ` +
      '400 "Extra inputs are not permitted". Moved to the request options. ' +
      'Fix the call site: client.messages.stream(params, { signal }).');
  }

  return anthropic.messages.stream(safeParams, safeOptions);
}

// ----- Simple response cache -----
// Keyed by the exact text sent in. If the same text comes in twice, we
// reuse the stored reply instead of calling the API again. This is just an
// in-memory Map, so it resets whenever the app restarts — that's fine for
// our purposes here.
const responseCache = new Map();

// Handles the 'ask-claude-start' request sent from preload.js's askClaude().
// This is the ONLY place the Anthropic API is called — entirely in the main
// process, so the API key never reaches the renderer or the page's JS.
// Unlike a plain request/response, this streams: as each piece of text
// arrives from Claude, we forward it to the renderer immediately via an
// 'ask-claude-chunk' event, then send one final 'ask-claude-done' event.
ipcMain.on('ask-claude-start', async (event, stuckText) => {
  const sender = event.sender;

  if (!anthropic) {
    // Shouldn't normally happen — the main window only opens once we have a
    // key — but guard against it rather than crash on a null client.
    sender.send('ask-claude-done', { ok: false, error: 'No API key is configured yet.' });
    return;
  }

  // Cache hit: skip the API call entirely and send the whole answer at once.
  const cached = responseCache.get(stuckText);
  if (cached) {
    sender.send('ask-claude-done', { ok: true, text: cached });
    return;
  }

  try {
    let fullText = '';

    const stream = streamMessage('text', {
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 300,
      system: SYSTEM_PROMPT,
      messages: [
        {
          role: 'user',
          content:
            `A user has been stuck for a while on the following problem. ` +
            `Give a short, encouraging, concrete suggestion for what to try next.\n\n${stuckText}`,
        },
      ],
    });

    // Fires once per piece of text as Claude generates it — this is what
    // makes the panel fill in word by word instead of appearing all at once.
    stream.on('text', (textDelta) => {
      fullText += textDelta;
      sender.send('ask-claude-chunk', textDelta);
    });

    // Waits for the stream to finish and assembles the complete message.
    await stream.finalMessage();

    // Only cache successful replies — never an error message.
    responseCache.set(stuckText, fullText);
    sender.send('ask-claude-done', { ok: true, text: fullText });
  } catch (error) {
    // Most-specific error type first, so the panel can show something
    // readable instead of a raw stack trace.
    let message = 'Something went wrong talking to Claude.';
    if (error instanceof Anthropic.AuthenticationError) {
      message = app.isPackaged
        ? 'Invalid API key. Quit IRIS, delete the file at ~/Library/Application Support/IRIS/iris-secret.enc, then reopen the app to re-enter it.'
        : 'Invalid or missing API key. Check ANTHROPIC_API_KEY in your .env file.';
    } else if (error instanceof Anthropic.RateLimitError) {
      message = 'Rate limited by the API. Try again in a moment.';
    } else if (error instanceof Anthropic.APIConnectionError) {
      message = 'Could not connect to the Anthropic API. Check your internet connection.';
    } else if (error instanceof Anthropic.APIError) {
      message = `API error: ${error.message}`;
    } else if (error.message) {
      message = error.message;
    }
    sender.send('ask-claude-done', { ok: false, error: message });
  }
});

// ===== Capturing what the reader is stuck on =====
//
// When the orb goes STUCK and the reader asks for help, we grab the part of
// the screen they have been re-reading and send it to Claude as an image,
// instead of the hardcoded paragraph this used to send.
//
// The gaze pipeline is not involved in any of this: it hands over one
// coordinate that it had already computed, and everything below is ordinary
// screen capture and an API call.

// The crop is in DIP (the same units as the gaze coordinates), sized to take
// in the surrounding paragraph rather than just the fixated word — the
// context above and below is most of what makes the excerpt explainable.
const GAZE_CAPTURE_WIDTH = 800;
const GAZE_CAPTURE_HEIGHT = 500;

// Anything larger than this gets resized server-side anyway, so sending more
// pixels costs upload time and buys nothing.
const CLAUDE_IMAGE_MAX_EDGE = 1568;

// What gets stored alongside the struggle event. Deliberately tiny: the event
// log lives in localStorage, which is a few megabytes in total, so persisting
// full-resolution PNGs would blow the quota within a session.
const CAPTURE_THUMBNAIL_WIDTH = 320;
const CAPTURE_THUMBNAIL_QUALITY = 60;

// How flat an image has to be before we call it blank. A real screenshot of
// anything — even a mostly-white page — spans far more than this.
const BLANK_CAPTURE_RANGE = 12;

// Samples the image sparsely and reports how much its pixels vary. Sparse on
// purpose: this runs on the interaction path, and a few thousand samples
// answers "is there anything here at all" just as well as reading every pixel.
function describeImageContent(image) {
  const bitmap = image.toBitmap(); // BGRA, 4 bytes per pixel
  let min = 255;
  let max = 0;
  let sum = 0;
  let samples = 0;

  // A prime-ish stride so the samples cut across the image rather than
  // landing in one column of a fixed-width row.
  const stride = 4 * 499;
  for (let i = 0; i + 2 < bitmap.length; i += stride) {
    const value = (bitmap[i] + bitmap[i + 1] + bitmap[i + 2]) / 3;
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
    samples += 1;
  }

  if (samples === 0) return { min: 0, max: 0, mean: 0, range: 0, samples: 0 };
  return {
    min: Math.round(min),
    max: Math.round(max),
    mean: Math.round(sum / samples),
    range: Math.round(max - min),
    samples,
  };
}

// Writes the cropped image somewhere it can be opened and looked at, because
// "the capture succeeded" and "the capture contains what you were reading" are
// different claims and only the file settles the second one. Keeps the most
// recent few and deletes the rest, so this never becomes a disk leak.
const CAPTURE_HISTORY_LIMIT = 10;

function saveCaptureToDisk(pngBuffer) {
  try {
    const directory = path.join(app.getPath('userData'), 'captures');
    fs.mkdirSync(directory, { recursive: true });

    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filePath = path.join(directory, `capture-${stamp}.png`);
    fs.writeFileSync(filePath, pngBuffer);

    const existing = fs.readdirSync(directory)
      .filter((name) => name.startsWith('capture-') && name.endsWith('.png'))
      .sort();
    for (const stale of existing.slice(0, Math.max(0, existing.length - CAPTURE_HISTORY_LIMIT))) {
      fs.unlinkSync(path.join(directory, stale));
    }

    return filePath;
  } catch (error) {
    // Never let a diagnostic write break the thing it is diagnosing.
    console.warn('[capture] could not save the capture to disk:', error.message);
    return null;
  }
}

// Captures the screen around a gaze point and crops it.
//
// `viewportX`/`viewportY` are in the MAIN WINDOW's viewport coordinates —
// the space the gaze mapping was calibrated in — so the first job here is
// converting them to screen coordinates.
async function captureGazeRegion(viewportX, viewportY) {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return { ok: false, error: 'The main window is gone, so there is nothing to locate the gaze against.' };
  }

  // The reported permission status is logged as CONTEXT, never used as a gate.
  // It was a gate, and that was the bug: getMediaAccessStatus('screen') is not
  // reliable on macOS — the same binary reported 'denied' and then 'granted'
  // within one session with nothing changed — so refusing to even attempt the
  // capture produced a confident "grant Screen Recording permission" message
  // in cases where capture would have worked fine, and hid whatever the real
  // failure was. Try the capture; report what actually happens.
  const reportedStatus = process.platform === 'darwin'
    ? systemPreferences.getMediaAccessStatus('screen')
    : 'n/a';

  // getContentBounds() is the main window's viewport expressed in screen
  // coordinates, so adding the two converts one space to the other. Both are
  // DIP, so nothing needs scaling here. This stays correct while the window is
  // hidden the "still tracking" way (opacity 0), because the window keeps its
  // real bounds — which is exactly the case that matters, since that is when
  // you are reading something else on top of it.
  const content = mainWindow.getContentBounds();
  const screenX = content.x + viewportX;
  const screenY = content.y + viewportY;

  const display = screen.getDisplayNearestPoint({ x: Math.round(screenX), y: Math.round(screenY) });
  const scale = display.scaleFactor || 1;


  // Ask for the screen at its true pixel resolution rather than its DIP size.
  // On a Retina display that is the difference between text Claude can read
  // and text it cannot.
  const sources = await desktopCapturer.getSources({
    types: ['screen'],
    thumbnailSize: {
      width: Math.round(display.size.width * scale),
      height: Math.round(display.size.height * scale),
    },
  });

  const source = sources.find((candidate) => String(candidate.display_id) === String(display.id)) || sources[0];
  if (!source) {
    console.error('[capture] FAILED: desktopCapturer returned no screen sources at all.');
    return { ok: false, stage: 'capture', error: 'Screen capture returned no sources.' };
  }
  if (source.thumbnail.isEmpty()) {
    console.error('[capture] FAILED: the screen source came back with an empty thumbnail.');
    return { ok: false, stage: 'capture', error: 'Screen capture returned an empty image.' };
  }

  // Crop box in DIP, centred on the gaze and clamped so it stays on the
  // display even when you are reading near an edge. Clamping the BOX (rather
  // than letting it run off and produce a smaller crop) keeps the excerpt the
  // same size wherever you look.
  const localX = screenX - display.bounds.x;
  const localY = screenY - display.bounds.y;
  const cropWidth = Math.min(GAZE_CAPTURE_WIDTH, display.size.width);
  const cropHeight = Math.min(GAZE_CAPTURE_HEIGHT, display.size.height);
  const cropX = Math.round(Math.min(Math.max(localX - cropWidth / 2, 0), display.size.width - cropWidth));
  const cropY = Math.round(Math.min(Math.max(localY - cropHeight / 2, 0), display.size.height - cropHeight));

  // crop() works in the captured image's own pixels, which is the DIP box
  // multiplied by the display's scale factor.
  let image = source.thumbnail.crop({
    x: Math.round(cropX * scale),
    y: Math.round(cropY * scale),
    width: Math.round(cropWidth * scale),
    height: Math.round(cropHeight * scale),
  });

  if (image.getSize().width > CLAUDE_IMAGE_MAX_EDGE) {
    image = image.resize({ width: CLAUDE_IMAGE_MAX_EDGE });
  }

  const finalSize = image.getSize();
  const pngBuffer = image.toPNG();

  // Is this a real screenshot, or a uniform placeholder? THIS is the honest
  // signal that screen recording is actually blocked: macOS hands back an
  // image of the right dimensions rather than an error, so dimensions alone
  // prove nothing. A flat single-colour frame is what a blocked capture looks
  // like — and unlike the permission status, it is measured, not reported.
  const content_ = describeImageContent(image);
  const savedPath = saveCaptureToDisk(pngBuffer);

  if (content_.range <= BLANK_CAPTURE_RANGE) {
    console.error('[capture] FAILED: the image is a flat single colour — the screen was captured but its contents were withheld.');
    return {
      ok: false,
      stage: 'capture',
      error: 'The screenshot came back blank. That usually means Screen Recording permission is not actually in effect for this binary — check System Settings > Privacy & Security > Screen Recording, and note that in development the entry is "Electron", not "IRIS".',
      savedPath,
    };
  }

  return {
    ok: true,
    savedPath,
    // Where the gaze actually sat inside the crop, as fractions of the
    // image's width and height. The crop is deliberately wider than the thing
    // being read so the model has surrounding context, which leaves it free to
    // explain the wrong part of the picture — this is how the prompt points it
    // back at the spot the reader was stuck on.
    focus: {
      xFraction: Math.min(Math.max((localX - cropX) / cropWidth, 0), 1),
      yFraction: Math.min(Math.max((localY - cropY) / cropHeight, 0), 1),
    },
    // Full resolution, for the API call only — this never leaves the main process.
    base64: pngBuffer.toString('base64'),
    // What actually gets stored on the event: the region in screen
    // coordinates, plus a small JPEG so the capture is reviewable later
    // without the event log growing by megabytes per struggle.
    region: {
      x: Math.round(display.bounds.x + cropX),
      y: Math.round(display.bounds.y + cropY),
      w: cropWidth,
      h: cropHeight,
    },
    thumbnail: 'data:image/jpeg;base64,' +
      image.resize({ width: CAPTURE_THUMBNAIL_WIDTH }).toJPEG(CAPTURE_THUMBNAIL_QUALITY).toString('base64'),
  };
}

// ----- What app is this being read in? -----
// Sent to Claude as a hint about what kind of thing the excerpt is from — a
// PDF reader, a browser, an editor — which changes how an ambiguous excerpt
// should be read.
//
// There is no cross-platform Electron API for another app's window title, so
// this is macOS-only via AppleScript and resolves to null everywhere else.
// Reading the window NAME needs Accessibility permission; the app name does
// not, so the script falls back to the app name alone rather than failing
// outright, and the whole thing resolves to null if System Events is blocked.
function getActiveWindowTitle() {
  if (process.platform !== 'darwin') return Promise.resolve(null);

  const script = `
    tell application "System Events"
      set frontApp to first application process whose frontmost is true
      set appName to name of frontApp
      try
        set windowName to name of first window of frontApp
      on error
        set windowName to ""
      end try
    end tell
    if windowName is "" then
      return appName
    else
      return appName & " — " & windowName
    end if
  `;

  return new Promise((resolve) => {
    // Timed out rather than awaited indefinitely: this is a nice-to-have hint,
    // and it must never be able to hold up the reply.
    execFile('osascript', ['-e', script], { timeout: 2000 }, (error, stdout) => {
      if (error) {
        console.warn('[capture] could not read the active window title:', error.message);
        resolve(null);
        return;
      }
      const title = String(stdout).trim();
      resolve(title || null);
    });
  });
}

// ----- The model -----
// Haiku, and a hard ceiling of 150 tokens: the answer is two short sentences
// arriving over someone's reading, so how fast the first words appear matters
// more than anything a larger model would add. Streamed, so they appear as
// they are written rather than all at once at the end.
const SCREEN_MODEL = 'claude-haiku-4-5-20251001';
// Raised from 150 now that the reply carries a topic label and a revision
// question alongside the explanation, plus the JSON envelope around all three.
// The explanation itself is still held to two sentences by the prompt.
const SCREEN_MAX_TOKENS = 300;

// ----- The prompt -----
// Stated once here rather than rebuilt per request. It works hardest at the
// three failure modes of showing a model a wide crop: opening by narrating the
// document back to a reader who is looking straight at it, explaining the
// context instead of the one spot they are stuck on, and inventing the
// surrounding text the crop cut away.
const SCREEN_SYSTEM_PROMPT =
  'You explain one thing to a reader who just got stuck on it. Answer in at ' +
  'most 2 short sentences, roughly 40 words, in plain language. ' +
  'Start immediately with the explanation. Never open by describing the ' +
  'situation, the document, or what the reader is doing — no "this is from a ' +
  'document about", no "it looks like you are reading". No preamble, no ' +
  'lists, no sign-off. ' +
  'The screenshot is a wide crop taken for context, but only the small area ' +
  'named in the message is what the reader is stuck on: use the rest purely ' +
  'to understand that area, and explain nothing else. ' +
  'Everything above and below the crop is missing, so if the area cannot be ' +
  'explained honestly from what is visible, say in one sentence what is ' +
  'missing rather than inventing it. ' +
  '\n\n' +
  'Reply with a single JSON object and nothing else — no code fence, no text ' +
  'before or after it — with exactly these three keys, in this order:\n' +
  '  "topic": 2-4 words naming what this is about, as a subject label rather ' +
  'than a sentence. Examples: "recursion", "Socratic method", "pointer ' +
  'arithmetic", "supply and demand".\n' +
  '  "explanation": the explanation described above.\n' +
  '  "question": one question that checks whether the reader can now do the ' +
  'thing the explanation just taught, asked about the specific material in ' +
  'front of them rather than about the topic in general. It is asked days ' +
  'later with only the topic and this snippet for context, so it must stand ' +
  'on its own — never "the above", "this code" or "as we discussed".';

// Describes WHERE in the image the reader was looking, in words, since the
// image itself carries no marker. Thirds rather than percentages: gaze is not
// pixel-precise, and a coarse description is one the model can actually act on.
function describeFocusArea(focus) {
  if (!focus) return 'the middle of the image';
  const band = (f, low, mid, high) => (f < 0.34 ? low : f < 0.67 ? mid : high);
  const vertical = band(focus.yFraction, 'upper', 'middle', 'lower');
  const horizontal = band(focus.xFraction, 'left', 'centre', 'right');
  return `the ${vertical} ${horizontal} part of the image`;
}

function buildScreenPrompt(windowTitle, focus) {
  return [
    `The reader has been re-reading ${describeFocusArea(focus)} and is stuck on it.`,
    'The rest of the screenshot is context only — do not explain it.',
    windowTitle ? `Application: ${windowTitle}` : null,
    'Explain that part now, directly, in at most 2 short sentences.',
    'Reply with the JSON object described in the system prompt: {"topic", "explanation", "question"}.',
  ].filter(Boolean).join('\n\n');
}


// ----- Reading the JSON reply as it streams -----
// The model now answers with {topic, explanation, question}, but the card
// still has to fill in word by word — waiting for the closing brace would undo
// the whole point of streaming. This walks the partial JSON as it arrives and
// yields only the newly-decoded characters of the "explanation" value, so what
// reaches the card is prose, never braces or key names.
function createExplanationStreamer() {
  let raw = '';
  let valueStart = -1;   // index in raw of the first character INSIDE the string
  let emitted = '';      // the decoded explanation emitted so far

  return function push(chunk) {
    raw += chunk;

    if (valueStart === -1) {
      const key = raw.indexOf('"explanation"');
      if (key === -1) return '';
      // Step past the key, its colon and any whitespace, to the opening quote.
      const quote = raw.indexOf('"', key + '"explanation"'.length);
      if (quote === -1) return '';
      valueStart = quote + 1;
    }

    // Decode from the start of the value each time rather than incrementally:
    // a chunk can split an escape sequence, and re-decoding a couple of
    // hundred characters per chunk costs nothing.
    let decoded = '';
    for (let i = valueStart; i < raw.length; i++) {
      const ch = raw[i];
      if (ch === '\\') {
        const next = raw[i + 1];
        if (next === undefined) break; // escape split across chunks — wait
        if (next === 'n') decoded += '\n';
        else if (next === 't') decoded += '\t';
        else if (next === 'u') {
          if (i + 5 >= raw.length) break;
          decoded += String.fromCharCode(parseInt(raw.slice(i + 2, i + 6), 16));
          i += 4;
        } else decoded += next;
        i += 1;
        continue;
      }
      if (ch === '"') break; // end of the explanation value
      decoded += ch;
    }

    if (decoded.length <= emitted.length) return '';
    const delta = decoded.slice(emitted.length);
    emitted = decoded;
    return delta;
  };
}

// The finished reply. Tolerant on purpose: a model told not to use a code
// fence still occasionally does, and a reply that cannot be parsed at all is
// far better shown as plain text than thrown away.
function parseScreenReply(text) {
  const trimmed = String(text || '').trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/```$/, '')
    .trim();

  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === 'object') {
      return {
        topic: typeof parsed.topic === 'string' ? parsed.topic.trim() : null,
        explanation: typeof parsed.explanation === 'string' ? parsed.explanation.trim() : trimmed,
        question: typeof parsed.question === 'string' ? parsed.question.trim() : null,
      };
    }
  } catch (error) {
    console.warn('[claude] reply was not valid JSON, using it as plain text:', error.message);
  }

  return { topic: null, explanation: trimmed, question: null };
}

// Handles 'ask-claude-screen-start'. Same streaming shape as the text-only
// 'ask-claude-start' handler above — chunks as they arrive, then one final
// message — but with a screenshot as the input and no response cache, since
// every capture is different by definition.
// The stream currently in flight, so a dismissal can abort it. One at a time —
// a second offer cannot be accepted while the first card is still open.
let screenRequestController = null;

// Sent by the renderer when the reader dismisses at any stage. Aborting the
// stream stops the tokens being generated and makes the pending
// finalMessage() reject, which the catch below recognises and reports as a
// cancellation rather than as a failure.
ipcMain.on('ask-claude-screen-cancel', () => {
  if (!screenRequestController) {
    console.log('[claude] cancel requested — nothing in flight');
    return;
  }
  console.log('[claude] cancel requested — aborting the in-flight request');
  screenRequestController.abort();
  screenRequestController = null;
});

ipcMain.on('ask-claude-screen-start', async (event, point) => {
  const sender = event.sender;
  const fail = (error) => sender.send('ask-claude-screen-done', { ok: false, error });

  if (!anthropic) {
    console.error('[claude] no client — the API key never loaded. ' +
      (app.isPackaged ? 'No stored key.' : 'Check ANTHROPIC_API_KEY in .env.'));
    fail('No API key is configured yet.');
    return;
  }

  let capture;
  try {
    capture = await captureGazeRegion(point.x, point.y);
  } catch (error) {
    console.error('[capture] threw:', error);
    fail('Screen capture failed: ' + (error.message || 'unknown error'));
    return;
  }
  if (!capture.ok) {
    // Labelled by STAGE. This is a screen-capture failure and must never be
    // reported as "couldn't reach Claude" — the API has not been called yet.
    fail('Screen capture failed: ' + capture.error);
    return;
  }

  const windowTitle = await getActiveWindowTitle();

  // Everything the renderer is allowed to keep. The full-resolution image is
  // deliberately not in here — it goes to the API and is then dropped.
  const captureForEvent = {
    region: capture.region,
    thumbnail: capture.thumbnail,
    windowTitle,
    capturedAt: Date.now(),
  };

  const promptText = buildScreenPrompt(windowTitle, capture.focus);

  const startedAt = Date.now();
  const controller = new AbortController();
  screenRequestController = controller;

  try {
    let fullText = '';
    // Only the explanation reaches the card; the topic and the question are
    // metadata and go back with the final message instead.
    const pushExplanation = createExplanationStreamer();

    // Plain messages.stream, not beta.messages.stream. This call used to carry
    // betas: ['server-side-fallback-2026-07-01'] and fallbacks: 'default',
    // which were added while the model was Opus and left in place when it
    // changed to Haiku. A beta header the chosen model does not accept is
    // rejected before the request is even looked at, which is a 400 on every
    // single call — the symptom being reported. There is nothing else in this
    // request that needs the beta endpoint.
    // Two arguments, and the split matters — see streamMessage above, which
    // is where the body and the options are separated and printed.
    const stream = streamMessage('screen', {
      model: SCREEN_MODEL,
      max_tokens: SCREEN_MAX_TOKENS,
      system: SCREEN_SYSTEM_PROMPT,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image',
              source: { type: 'base64', media_type: 'image/png', data: capture.base64 },
            },
            { type: 'text', text: promptText },
          ],
        },
      ],
    }, { signal: controller.signal });

    stream.on('text', (textDelta) => {
      fullText += textDelta;
      const visible = pushExplanation(textDelta);
      if (visible) sender.send('ask-claude-screen-chunk', visible);
    });

    const finalMessage = await stream.finalMessage();

    // A refusal here means the fallback chain declined too. It arrives as a
    // normal 200 response, not a thrown error, so it has to be checked
    // explicitly or it would surface as an empty reply.
    if (finalMessage.stop_reason === 'refusal') {
      fail('Claude declined to describe this screenshot.');
      return;
    }

    const reply = parseScreenReply(fullText);

    screenRequestController = null;

    sender.send('ask-claude-screen-done', {
      ok: true,
      // `text` stays the explanation, which is what every existing caller
      // means by it — the card shows it and the event stores it as helpText.
      text: reply.explanation,
      topic: reply.topic,
      question: reply.question,
      capture: captureForEvent,
    });
  } catch (error) {
    screenRequestController = null;

    // A dismissal is not a failure. The renderer has already taken the card
    // down, so this reply is only closing the promise out.
    if (controller.signal.aborted || (error && error.name === 'AbortError')) {
      console.log(`[claude] request cancelled by the reader after ${Date.now() - startedAt}ms`);
      sender.send('ask-claude-screen-done', { ok: false, cancelled: true, error: 'Cancelled.', capture: captureForEvent });
      return;
    }

    // The API's OWN words, not a rewrite of them. The advice that used to
    // replace the message on a 401 was hiding the one line that says what is
    // actually wrong — "credit balance is too low" and "invalid x-api-key" are
    // both 401s, and only one of them is fixed by re-entering the key.
    const status = error && error.status !== undefined ? `HTTP ${error.status}` : 'no HTTP response';
    const apiMessage =
      (error && error.error && error.error.error && error.error.error.message) ||
      (error && error.error && error.error.message) ||
      (error && error.message) ||
      'unknown error';
    const apiType =
      (error && error.error && error.error.error && error.error.error.type) ||
      (error && error.error && error.error.type) ||
      (error && error.constructor && error.constructor.name) ||
      'error';

    let message = `Claude API (${status}, ${apiType}): ${apiMessage}`;
    // Appended, never substituted — the advice comes after the real error.
    if (error instanceof Anthropic.AuthenticationError) {
      message += app.isPackaged
        ? ' — quit IRIS, delete ~/Library/Application Support/IRIS/iris-secret.enc, then reopen to re-enter the key.'
        : ' — check ANTHROPIC_API_KEY in your .env file.';
    } else if (error instanceof Anthropic.APIConnectionError) {
      message += ' — check your internet connection.';
    }
    console.error(`[claude] request failed after ${Date.now() - startedAt}ms: ${message}`);

    // The capture still goes back on a failure, so the event records what was
    // on screen even when the explanation never arrived.
    sender.send('ask-claude-screen-done', { ok: false, error: message, capture: captureForEvent });
  }
});
