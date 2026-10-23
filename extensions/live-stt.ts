/**
 * Live voice dictation for the pi TUI, powered by a Qwen3-ASR websocket server.
 *
 * Hold Space on an empty prompt and talk; the transcript appears live in the editor.
 * Release to stop. A short tap latches recording instead; tap again to stop.
 *
 *   hold Space   record while held (release is detected when key auto-repeat stops)
 *   tap Space    latch recording, tap Space again to stop
 *   Enter        while recording: stop and send
 *   Esc          cancel and restore the prompt
 *   Space again  while the prompt still holds untouched dictation: dictate more and append
 *
 * Audio source (config "micSource"):
 *   "local"   this machine records (parec / pw-record / arecord / ffmpeg) and streams 16 kHz mono
 *             s16le PCM to the server. Over ssh the mic can come from a forwarded PulseAudio socket.
 *   "server"  the STT server records its own microphone (needs `--mic-allow` on the server).
 *             Nothing to install here; use it when the mic and the server share a machine.
 *
 * Config (all optional): ~/.pi/agent/live-stt.json
 *   { "url", "trigger", "micSource", "recorder": ["cmd", ...args], "holdDetectMs", "releaseMs" }
 * Change the server from inside pi: `/live-stt url 192.168.1.20` (or host:port, or a full ws:// url);
 * it applies immediately and is saved to the config file.
 * Env overrides: PI_LIVE_STT_URL, PI_LIVE_STT_TRIGGER, PI_LIVE_STT_MIC, PI_LIVE_STT_REC
 *   (PI_LIVE_STT_REC is a shell command that prints raw PCM, handy for testing).
 */
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { CustomEditor, type ExtensionAPI, type ExtensionContext } from "@earendil-works/pi-coding-agent";
import { matchesKey, truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";

interface Config {
	url: string;
	trigger: string;
	recorder?: string[];
	/** "local": record on this machine and stream PCM. "server": the STT server records its own mic (no recorder needed here). */
	micSource: "local" | "server";
	/** Push-to-talk: a repeat of the trigger within this window after the first press means "held" (> OS key-repeat delay). */
	holdDetectMs: number;
	/** Held mode ends when no trigger repeat arrives for this long (> OS key-repeat interval). */
	releaseMs: number;
}

const RECORDERS: string[][] = [
	["parec", "--rate=16000", "--channels=1", "--format=s16le", "--raw"],
	["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16", "-"],
	["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
	["ffmpeg", "-loglevel", "error", "-f", "pulse", "-i", "default", "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
];
const CONFIG_FILE = join(homedir(), ".pi", "agent", "live-stt.json");
const CONNECT_TIMEOUT_MS = 5000;
const NO_AUDIO_TIMEOUT_MS = 4000;
const FINISH_TIMEOUT_MS = 15000;

function loadConfig(): Config {
	const cfg: Config = {
		url: "ws://192.168.7.214:8765/v1/stt?sample_rate=16000&encoding=pcm",
		trigger: "space",
		micSource: "local",
		holdDetectMs: 800,
		releaseMs: 250,
	};
	if (existsSync(CONFIG_FILE)) {
		try {
			Object.assign(cfg, JSON.parse(readFileSync(CONFIG_FILE, "utf8")));
		} catch (error) {
			console.warn(`live-stt: ignoring bad ${CONFIG_FILE}: ${error}`);
		}
	}
	if (process.env.PI_LIVE_STT_URL) cfg.url = process.env.PI_LIVE_STT_URL;
	if (process.env.PI_LIVE_STT_TRIGGER) cfg.trigger = process.env.PI_LIVE_STT_TRIGGER;
	if (process.env.PI_LIVE_STT_MIC === "server" || process.env.PI_LIVE_STT_MIC === "local") cfg.micSource = process.env.PI_LIVE_STT_MIC;
	if (process.env.PI_LIVE_STT_REC) cfg.recorder = ["sh", "-c", process.env.PI_LIVE_STT_REC];
	return cfg;
}

/** "host", "host:port" or a full ws(s):// url -> STT websocket url. */
function toUrl(input: string): string {
	if (/^wss?:\/\//.test(input)) return input;
	const hostPort = input.includes(":") ? input : `${input}:8765`;
	return `ws://${hostPort}/v1/stt?sample_rate=16000&encoding=pcm`;
}

/** Persist the url into the config file, keeping its other keys. */
function saveUrl(url: string): void {
	let saved: Record<string, unknown> = {};
	if (existsSync(CONFIG_FILE)) saved = JSON.parse(readFileSync(CONFIG_FILE, "utf8"));
	mkdirSync(dirname(CONFIG_FILE), { recursive: true });
	writeFileSync(CONFIG_FILE, `${JSON.stringify({ ...saved, url }, null, 2)}\n`);
}

function pickRecorder(cfg: Config): string[] {
	if (cfg.recorder?.length) return cfg.recorder;
	const found = RECORDERS.find(([bin]) => spawnSync("which", [bin], { stdio: "ignore" }).status === 0);
	if (!found) throw new Error("no recorder on PATH (install pulseaudio-utils for parec, or alsa-utils for arecord)");
	return found;
}

/** One dictation: recorder -> websocket -> live text callbacks. */
class Dictation {
	private ws?: WebSocket;
	private proc?: ChildProcess;
	private text = "";
	private finished = false;
	private resolveDone?: (text: string) => void;
	private stderr = "";

	constructor(
		private cfg: Config,
		private onText: (text: string) => void,
		private onError: (message: string) => void,
	) {}

	async start(): Promise<void> {
		const serverMic = this.cfg.micSource === "server";
		const recorder = serverMic ? [] : pickRecorder(this.cfg);
		await this.connect(serverMic);
		if (serverMic) return;
		const [bin, ...args] = recorder;
		const proc = spawn(bin!, args, { stdio: ["ignore", "pipe", "pipe"] });
		this.proc = proc;
		let gotAudio = false;
		const noAudio = setTimeout(() => {
			if (!gotAudio) this.fail(`no audio from ${bin} in ${NO_AUDIO_TIMEOUT_MS / 1000}s (is the mic reachable?) ${this.stderr.trim()}`);
		}, NO_AUDIO_TIMEOUT_MS);
		proc.stdout!.on("data", (chunk: Buffer) => {
			gotAudio = true;
			if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(chunk);
		});
		proc.stderr!.on("data", (chunk: Buffer) => (this.stderr += chunk.toString()));
		proc.on("error", (error) => this.fail(`${bin}: ${error.message}`));
		proc.on("exit", () => clearTimeout(noAudio));
	}

	private connect(serverMic: boolean): Promise<void> {
		return new Promise((resolve, reject) => {
			const ws = new WebSocket(serverMic ? `${this.cfg.url}${this.cfg.url.includes("?") ? "&" : "?"}mic=server` : this.cfg.url);
			this.ws = ws;
			const timer = setTimeout(() => reject(new Error(`STT server not reachable at ${this.cfg.url}`)), CONNECT_TIMEOUT_MS);
			ws.onerror = () => {
				clearTimeout(timer);
				reject(new Error(`cannot connect to STT server ${this.cfg.url} (server running? ssh RemoteForward 8765?)`));
			};
			ws.onclose = () => this.complete();
			ws.onmessage = (event) => {
				let msg: { type?: string; text?: string; message?: string };
				try {
					msg = JSON.parse(String(event.data));
				} catch {
					return;
				}
				if (msg.type === "transcript.created") {
					clearTimeout(timer);
					resolve();
				} else if (msg.type === "error") {
					clearTimeout(timer);
					this.finished = true;
					reject(new Error(msg.message ?? "server error"));
				} else if (msg.type === "transcript.partial" && !this.finished) {
					this.text = msg.text ?? this.text;
					this.onText(this.text);
				} else if (msg.type === "transcript.done") {
					this.text = msg.text ?? this.text;
					this.complete();
				}
			};
		});
	}

	/** Stop the mic, flush the server, resolve with the final text. */
	stop(): Promise<string> {
		this.killRecorder();
		return new Promise((resolve) => {
			this.resolveDone = resolve;
			if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: "audio.done" }));
			else return this.complete();
			setTimeout(() => this.complete(), FINISH_TIMEOUT_MS);
		});
	}

	cancel(): void {
		this.finished = true;
		this.killRecorder();
		this.ws?.close();
	}

	private fail(message: string): void {
		if (this.finished) return;
		this.cancel();
		this.onError(message);
	}

	private complete(): void {
		if (this.finished && !this.resolveDone) return;
		this.finished = true;
		this.killRecorder();
		this.ws?.close();
		this.resolveDone?.(this.text);
		this.resolveDone = undefined;
	}

	private killRecorder(): void {
		this.proc?.kill("SIGTERM");
		this.proc = undefined;
	}
}

type Mode = "idle" | "recording" | "transcribing";
/** pending: just pressed, tap or hold not known yet; holding: key repeat seen (push-to-talk); latched: tapped, press again to stop. */
type Hold = "pending" | "holding" | "latched";
const SUPPRESS_AFTER_STOP_MS = 500; // swallow trailing key-repeat of the press that stopped us

function join2(base: string, text: string): string {
	return base + (base && text && !/\s$/.test(base) ? " " : "") + text;
}

class LiveEditor extends CustomEditor {
	private mode: Mode = "idle";
	private session?: Dictation;
	private base = "";
	private hold: Hold = "pending";
	private holdTimer?: ReturnType<typeof setTimeout>;
	private releaseTimer?: ReturnType<typeof setTimeout>;
	private suppressUntil = 0;
	/** Prompt text we inserted and the user has not touched; Space may append another dictation to it. */
	private untouched?: string;

	constructor(
		tui: ConstructorParameters<typeof CustomEditor>[0],
		theme: ConstructorParameters<typeof CustomEditor>[1],
		keybindings: ConstructorParameters<typeof CustomEditor>[2],
		private ctx: ExtensionContext,
		private cfg: Config,
		private send: (text: string) => void,
	) {
		super(tui, theme, keybindings);
	}

	handleInput(data: string): void {
		const isTrigger = matchesKey(data, this.cfg.trigger as never);
		if (this.mode === "idle") {
			if (isTrigger && Date.now() < this.suppressUntil) return; // key-repeat tail of the press that stopped us
			const text = this.getText();
			const mayStart = this.cfg.trigger !== "space" || text === "" || text === this.untouched;
			if (isTrigger && mayStart && !this.isShowingAutocomplete()) {
				void this.begin();
				return;
			}
			this.untouched = undefined; // any other key edits the prompt
			super.handleInput(data);
			return;
		}
		if (matchesKey(data, "escape")) return this.cancel();
		if (this.mode === "transcribing") return;
		if (isTrigger) return this.onTriggerWhileRecording();
		if (matchesKey(data, "enter")) return void this.finish(true);
		if (data.length === 1 && data.charCodeAt(0) >= 32) return; // typing would be overwritten by the next partial
		super.handleInput(data);
	}

	/** First press while pending is a repeat (held); repeats keep holding alive; a press after a tap stops. */
	private onTriggerWhileRecording(): void {
		if (this.hold === "latched") return void this.finish(false);
		this.hold = "holding";
		clearTimeout(this.holdTimer);
		clearTimeout(this.releaseTimer);
		this.releaseTimer = setTimeout(() => void this.finish(false), this.cfg.releaseMs);
		this.tui.requestRender();
	}

	private clearTimers(): void {
		clearTimeout(this.holdTimer);
		clearTimeout(this.releaseTimer);
	}

	private show(text: string): void {
		this.setText(join2(this.base, text));
		this.tui.requestRender();
	}

	private async begin(): Promise<void> {
		this.base = this.getText();
		this.mode = "recording";
		this.hold = "pending";
		this.untouched = undefined;
		this.holdTimer = setTimeout(() => {
			if (this.hold === "pending") {
				this.hold = "latched";
				this.tui.requestRender();
			}
		}, this.cfg.holdDetectMs);
		this.tui.requestRender();
		const session = new Dictation(
			this.cfg,
			(text) => this.show(text),
			(message) => this.abort(message),
		);
		this.session = session;
		try {
			await session.start();
		} catch (error) {
			this.abort(error instanceof Error ? error.message : String(error));
		}
	}

	private abort(message: string): void {
		this.clearTimers();
		this.session?.cancel();
		this.session = undefined;
		this.mode = "idle";
		this.setText(this.base);
		this.tui.requestRender();
		this.ctx.ui.notify(`live-stt: ${message}`, "error");
	}

	cancel(): void {
		this.clearTimers();
		this.suppressUntil = Date.now() + SUPPRESS_AFTER_STOP_MS;
		this.session?.cancel();
		this.session = undefined;
		this.mode = "idle";
		this.setText(this.base);
		this.tui.requestRender();
	}

	private async finish(submit: boolean): Promise<void> {
		const session = this.session;
		if (!session) return;
		this.clearTimers();
		this.suppressUntil = Date.now() + SUPPRESS_AFTER_STOP_MS;
		this.mode = "transcribing";
		this.tui.requestRender();
		const text = (await session.stop()).trim();
		if (this.session !== session) return; // cancelled meanwhile
		this.session = undefined;
		this.mode = "idle";
		this.suppressUntil = Date.now() + SUPPRESS_AFTER_STOP_MS;
		this.setText(join2(this.base, text) + (text && !submit ? " " : ""));
		this.untouched = this.getText();
		this.tui.requestRender();
		if (submit) {
			const prompt = this.getText().trim();
			if (prompt) {
				this.setText("");
				this.send(prompt);
			}
		}
	}

	dispose(): void {
		this.clearTimers();
		this.session?.cancel();
	}

	render(width: number): string[] {
		const lines = super.render(width);
		if (this.mode === "idle" || lines.length === 0) return lines;
		const hint = this.hold === "holding" ? "release to stop" : "tap to stop · enter send · esc cancel";
		const label =
			this.mode === "recording" ? ` \x1b[31m●\x1b[0m recording  ${hint} ` : " \x1b[33m•\x1b[0m transcribing ";
		const last = lines.length - 1;
		if (visibleWidth(lines[last]!) > visibleWidth(label)) {
			lines[last] = truncateToWidth(lines[last]!, width - visibleWidth(label), "") + label;
		}
		return lines;
	}
}

export default function liveStt(pi: ExtensionAPI) {
	const cfg = loadConfig();
	let editor: LiveEditor | undefined;

	pi.on("session_start", (_event, ctx) => {
		if (!ctx.hasUI) return;
		ctx.ui.setEditorComponent((tui, theme, kb) => {
			editor = new LiveEditor(tui, theme, kb, ctx, cfg, (text) => {
				if (ctx.isIdle()) pi.sendUserMessage(text);
				else pi.sendUserMessage(text, { deliverAs: "followUp" });
			});
			return editor;
		});
	});

	pi.on("session_shutdown", () => editor?.dispose());

	pi.registerCommand("live-stt", {
		description: "Live dictation: status, doctor (check recorder and STT server), url <host[:port]> (change server).",
		handler: async (args, ctx) => {
			const [verb = "", target] = args.trim().split(/\s+/);
			const action = verb.toLowerCase() || "status";
			if (action === "url") {
				if (!target) {
					ctx.ui.notify(`live-stt url: ${cfg.url}`, "info");
					return;
				}
				cfg.url = toUrl(target);
				try {
					saveUrl(cfg.url);
				} catch (error) {
					ctx.ui.notify(`live-stt: url set for this session, not saved: ${error}`, "warning");
					return;
				}
				const shadow = process.env.PI_LIVE_STT_URL ? " (PI_LIVE_STT_URL will override it on restart)" : "";
				ctx.ui.notify(`live-stt url -> ${cfg.url}${shadow}. Try /live-stt doctor`, "info");
				return;
			}
			if (action === "status") {
				ctx.ui.notify(`live-stt: trigger=${cfg.trigger} mic=${cfg.micSource} url=${cfg.url} PULSE_SERVER=${process.env.PULSE_SERVER ?? "(unset)"}`, "info");
				return;
			}
			if (action !== "doctor") {
				ctx.ui.notify("Usage: /live-stt [status|doctor|url [host[:port]]]", "error");
				return;
			}
			try {
				const rec = cfg.micSource === "server" ? ["server-mic"] : pickRecorder(cfg);
				const server = await new Promise<string>((resolve, reject) => {
					const ws = new WebSocket(cfg.url);
					const timer = setTimeout(() => reject(new Error("timeout")), CONNECT_TIMEOUT_MS);
					ws.onmessage = () => {
						clearTimeout(timer);
						ws.close();
						resolve("reachable");
					};
					ws.onerror = () => {
						clearTimeout(timer);
						reject(new Error("connection refused"));
					};
				});
				ctx.ui.notify(`live-stt ok: recorder ${rec[0]}, STT server ${server} (${cfg.url})`, "info");
			} catch (error) {
				ctx.ui.notify(`live-stt problem: ${error instanceof Error ? error.message : error}`, "error");
			}
		},
	});
}
