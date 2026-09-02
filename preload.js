// The preload script runs in the renderer's context but with access to a
// few privileged Node/Electron APIs, BEFORE the page's own scripts run.
// Its only job here is to expose ONE safe function to the page — never the
// raw ipcRenderer, and never the Anthropic API key (which never leaves
// main.js). This keeps the page's normal JS sandboxed while still letting
// it ask the main process to talk to Claude on its behalf.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('iris', {
  // Starts a Claude request and streams the reply back piece by piece.
  // `onChunk` is called once per piece of text as it arrives, so the caller
  // can render it word-by-word. Resolves at the end with the full result:
  // either { ok: true, text } or { ok: false, error }.
  //
  // Only one of these is ever in flight at a time in this app (the "Yes,
  // help me" button is disabled while waiting), so we can use fixed channel
  // names below instead of tagging each request with a unique id.
  askClaude: (stuckText, onChunk) => {
    return new Promise((resolve) => {
      const handleChunk = (event, textDelta) => onChunk(textDelta);

      const handleDone = (event, result) => {
        // Stop listening once the request is finished, so a later request
        // doesn't end up with duplicate listeners piling up.
        ipcRenderer.removeListener('ask-claude-chunk', handleChunk);
        ipcRenderer.removeListener('ask-claude-done', handleDone);
        resolve(result);
      };

      ipcRenderer.on('ask-claude-chunk', handleChunk);
      ipcRenderer.on('ask-claude-done', handleDone);
      ipcRenderer.send('ask-claude-start', stuckText);
    });
  },

  // Used only by setup.html's one-time first-run screen (packaged builds).
  // Sends the pasted key to main.js to be encrypted and stored — the plain
  // text only ever exists in that one setup window, briefly, on this machine.
  saveApiKey: (key) => ipcRenderer.invoke('save-api-key', key),

  // "Is this window currently maximised (or full screen)?" — answered by
  // the main process, which is the only side that actually knows. Used to
  // block calibration in a non-maximised window.
  isWindowMaximized: () => ipcRenderer.invoke('is-window-maximized'),

  // Asks the main process to screenshot the region around a gaze point and
  // have Claude explain it. `point` is {x, y} in THIS window's viewport
  // coordinates — the space gaze predictions live in; main.js converts it to
  // screen coordinates, since only it knows where the window sits.
  //
  // Streams like askClaude() above, and resolves with
  // { ok, text, capture } or { ok: false, error, capture }. `capture` carries
  // the region, a small thumbnail and the active window title, for storing on
  // the struggle event — the full-resolution screenshot never leaves the main
  // process. It is present on failures too, so the event still records what
  // was on screen when the explanation could not be fetched.
  askClaudeAboutScreen: (point, onChunk) => {
    return new Promise((resolve) => {
      const handleChunk = (event, textDelta) => onChunk(textDelta);

      const handleDone = (event, result) => {
        ipcRenderer.removeListener('ask-claude-screen-chunk', handleChunk);
        ipcRenderer.removeListener('ask-claude-screen-done', handleDone);
        resolve(result);
      };

      ipcRenderer.on('ask-claude-screen-chunk', handleChunk);
      ipcRenderer.on('ask-claude-screen-done', handleDone);
      ipcRenderer.send('ask-claude-screen-start', point);
    });
  },

  // Abandons an in-flight askClaudeAboutScreen(). The main process aborts the
  // stream; the promise above still settles, with { ok: false, cancelled: true },
  // so no caller is left waiting on a request nobody wants any more.
  cancelClaudeAboutScreen: () => ipcRenderer.send('ask-claude-screen-cancel'),

  // ----- The help card in the orb window -----
  // Both halves live here because this same preload is loaded by both windows:
  // the main window drives the card, the orb window renders it and reports
  // back. See the "help card" section of main.js for the relay.

  // Main window -> orb card. `message` is {type, ...}:
  //   { type: 'show',   mode, body }  open it, growing out of the orb
  //   { type: 'update', mode, body }  new content (streaming text, mode change)
  //   { type: 'fade' }                start the slow being-ignored fade
  //   { type: 'hide' }                collapse it back into the orb
  // Fire-and-forget: the card must never be able to stall the caller, and
  // streaming an answer sends one of these per chunk.
  sendOrbCard: (message) => ipcRenderer.send('orb-card', message),

  // Orb window <- main window. Returns an unsubscribe function.
  onOrbCard: (callback) => {
    const handler = (event, message) => callback(message);
    ipcRenderer.on('orb-card', handler);
    return () => ipcRenderer.removeListener('orb-card', handler);
  },

  // Orb window -> main window: a card button was pressed.
  sendOrbCardAction: (action) => ipcRenderer.send('orb-card-action', action),
  onOrbCardAction: (callback) => {
    const handler = (event, action) => callback(action);
    ipcRenderer.on('orb-card-action', handler);
    return () => ipcRenderer.removeListener('orb-card-action', handler);
  },

  // Orb window -> main window: where the card is, so the main window (the only
  // one with gaze data) can tell whether it is being looked at. Sent in the orb
  // window's own client coordinates; main.js converts it into the main
  // window's viewport before forwarding. `null` clears it.
  sendOrbCardRect: (rect) => ipcRenderer.send('orb-card-rect', rect),
  onOrbCardRect: (callback) => {
    const handler = (event, rect) => callback(rect);
    ipcRenderer.on('orb-card-rect', handler);
    return () => ipcRenderer.removeListener('orb-card-rect', handler);
  },

  // Orb window -> main process: the pointer has crossed into (or out of) the
  // card, so the window should stop (or resume) being click-through.
  setOrbCardInteractive: (interactive) => ipcRenderer.send('orb-card-interactive', interactive),

  // Where the orb itself is, in THIS window's viewport coordinates. Needed
  // because during the offer stage there is no card — the red orb IS the
  // offer — so "ignoring it" has to be measured against the orb.
  getOrbRect: () => ipcRenderer.invoke('get-orb-rect'),

  // Turns the card's global keyboard fallbacks on and off. Driven by the
  // stuck moment's lifetime rather than the card's, because accept has to be
  // reachable during the offer stage, when no card exists yet.
  setHelpShortcuts: (active) => ipcRenderer.send('help-shortcuts', active),


  // A coarse 64x40 greyscale sample of the screen, for noticing a scroll or an
  // app switch. Too low-res to read anything from — see the handler in main.js.
  getScreenFingerprint: () => ipcRenderer.invoke('screen-fingerprint'),

  // ----- Calibration persistence -----
  // The fit is four numbers; the main process owns the file it lives in.
  saveCalibration: (payload) => ipcRenderer.invoke('save-calibration', payload),
  loadCalibration: () => ipcRenderer.invoke('load-calibration'),
  clearCalibration: () => ipcRenderer.invoke('clear-calibration'),

  // ----- Zone metric logging -----
  // The main process owns the file; this window owns the numbers and sends
  // finished rows. start() resolves with the path the CSV is being written to.
  startZoneLog: () => ipcRenderer.invoke('zone-log-start'),
  writeZoneLogRow: (row) => ipcRenderer.send('zone-log-row', row),
  stopZoneLog: () => ipcRenderer.invoke('zone-log-stop'),

  // The three global shortcuts that have to work while another app has focus.
  // They arrive here as plain notifications; this window decides what they do,
  // because it is the only side that knows the current numbers.
  onZoneCommand: (callback) => {
    const toggleLog = () => callback('toggle-log');
    const mark = () => callback('mark');
    const toggleReadout = () => callback('toggle-readout');
    ipcRenderer.on('zone-log-toggle', toggleLog);
    ipcRenderer.on('zone-log-mark', mark);
    ipcRenderer.on('zone-readout-toggle', toggleReadout);
    return () => {
      ipcRenderer.removeListener('zone-log-toggle', toggleLog);
      ipcRenderer.removeListener('zone-log-mark', mark);
      ipcRenderer.removeListener('zone-readout-toggle', toggleReadout);
    };
  },

  // The floating orb's optional numeric readout. Main window -> orb window.
  setOrbReadout: (payload) => ipcRenderer.send('orb-readout', payload),
  onOrbReadout: (callback) => {
    const handler = (event, payload) => callback(payload);
    ipcRenderer.on('orb-readout', handler);
    return () => ipcRenderer.removeListener('orb-readout', handler);
  },

  // ----- Orb window bridge -----
  // This same preload is loaded by BOTH windows, so both halves of the orb
  // relay live here: the main window uses setOrbState() to push, the orb
  // window uses onOrbState() to receive. Neither can reach the other
  // directly — the main process sits in between (see main.js).

  // Main window -> orb. Called from index.html's tick() when the attention
  // state changes. Fire-and-forget: there is nothing useful to wait for, and
  // the orb must never be able to stall the animation loop.
  setOrbState: (state) => ipcRenderer.send('orb-state', state),

  // ----- Which of the two orb looks to draw -----
  // Presentation only, and deliberately on its own channel rather than folded
  // into orb-state: the state channel fires from the gaze loop and must stay
  // exactly one send per state CHANGE, while this one fires only when someone
  // picks a different style in the dashboard.
  //
  // Main window -> orb. The main process also remembers the last value and
  // replays it whenever the orb window is created, so a restarted orb window
  // never comes up in the wrong style.
  setOrbStyle: (style) => ipcRenderer.send('orb-style', style),

  // Orb window -> its own renderer. Returns an unsubscribe function, matching
  // every other on* here.
  onOrbStyle: (callback) => {
    const handler = (event, style) => callback(style);
    ipcRenderer.on('orb-style', handler);
    return () => ipcRenderer.removeListener('orb-style', handler);
  },

  // Orb window <- main. Returns an unsubscribe function, so a caller that
  // ever needs to stop listening isn't forced to hold onto the raw handler.
  onOrbState: (callback) => {
    const handler = (event, state) => callback(state);
    ipcRenderer.on('orb-state', handler);
    return () => ipcRenderer.removeListener('orb-state', handler);
  },

  // ----- Showing and hiding either window -----
  // Each resolves to the resulting visibility (true = visible), so the caller
  // can update its own label or indicator without asking again.
  showOrb: () => ipcRenderer.invoke('set-orb-visible', true),
  hideOrb: () => ipcRenderer.invoke('set-orb-visible', false),
  toggleOrb: () => ipcRenderer.invoke('toggle-orb-visible'),
  isOrbVisible: () => ipcRenderer.invoke('is-orb-visible'),

  // hideMain() defaults to the "still tracking" hide (invisible but
  // composited, so the gaze loop keeps running). Pass { full: true } for a
  // real window hide, which removes it from the Dock/app switcher but pauses
  // tracking until it is shown again — see setMainWindowVisible() in main.js.
  showMain: () => ipcRenderer.invoke('set-main-visible', true),
  hideMain: (options) => ipcRenderer.invoke('set-main-visible', false, options),
  toggleMain: () => ipcRenderer.invoke('toggle-main-visible'),
  isMainVisible: () => ipcRenderer.invoke('is-main-visible'),
});
