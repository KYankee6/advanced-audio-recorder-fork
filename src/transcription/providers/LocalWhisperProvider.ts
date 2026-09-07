/**
 * Transcription via a local whisper.cpp binary. Desktop only - it shells
 * out through Node's child_process, which is unavailable in the mobile
 * app. The service hands this provider a single decoded WAV per request (it
 * declares no upload limit), which is written to a temp file, split into
 * independent CHUNK_SECONDS parts when long (whole-file runs loop on a
 * repeated phrase), transcribed to JSON, merged on the original timeline,
 * then all temp files are removed.
 *
 * The run is bounded the way a network request is, because it is the same kind
 * of thing: an external operation of unpredictable length. A limit on the
 * process is what a limit on a socket is elsewhere, and the run's cancel
 * reaches it the same way. Node's `execFile` takes both as options, so neither
 * needs a mechanism of its own.
 * @module transcription/providers/LocalWhisperProvider
 */

import {
	LOCAL_WHISPER_MAX_BUFFER_BYTES,
	PLUGIN_LOG_PREFIX,
	TRANSCRIPTION_PROVIDER_IDS,
} from '../../constants';
import { whisperPromptValue } from '../dictionaryBias';
import type { TranscriptSegment } from '../TranscriptTypes';
import { LOCAL_WHISPER_CAPABILITIES } from './capabilities';
import type { WhisperResult } from './whisperResponse';
import { isRecord, num } from './responseUtils';
import { randomToken } from '../../utils/ids';
import type {
	AudioPayload,
	ProviderCapabilities,
	TranscribeOptions,
	TranscriptionProvider,
} from './TranscriptionProvider';

/** How one child process is bounded and cancelled. */
interface ExecFileOptions {
	maxBuffer: number;
	/** Milliseconds after which Node kills the process. */
	timeout?: number;
	/** The run's cancel; aborting it kills the process. */
	signal?: AbortSignal;
}

/** Minimal Node surface used by the local provider. */
interface NodeModules {
	childProcess: {
		execFile: (
			file: string,
			args: string[],
			options: ExecFileOptions,
			callback: (
				error: Error | null,
				stdout: string,
				stderr: string,
			) => void,
		) => void;
	};
	fs: {
		writeFileSync: (path: string, data: Uint8Array | string) => void;
		readFileSync: {
			(path: string, encoding: string): string;
			(path: string): Uint8Array;
		};
		rmSync: (
			path: string,
			options: { force: boolean; recursive?: boolean },
		) => void;
		mkdirSync: (path: string) => void;
		readdirSync: (path: string) => string[];
		openSync: (path: string, flags: string) => number;
		readSync: (
			fd: number,
			buffer: Uint8Array,
			offset: number,
			length: number,
			position: number,
		) => number;
		closeSync: (fd: number) => void;
	};
	os: { tmpdir: () => string };
	path: { join: (...parts: string[]) => string };
}

/**
 * Length of one independent whisper run. A whole-meeting run accumulates
 * decoder context until it loops on a phrase ("네, 네, 네"); the verified fix
 * is short chunked runs with zero context (see runWhisper).
 */
export const LOCAL_WHISPER_CHUNK_SECONDS = 240;
/** Adjacent identical segments closer than this are decoder residue, not speech. */
export const DUPE_GAP_SECONDS = 5;

function readU32LE(bytes: Uint8Array, offset: number): number {
	return (
		(bytes[offset]! |
			(bytes[offset + 1]! << 8) |
			(bytes[offset + 2]! << 16) |
			(bytes[offset + 3]! << 24)) >>> 0
	);
}

/**
 * Duration of a canonical PCM WAV from its RIFF chunk walk, or null when the
 * layout is not parseable (callers then skip chunking and run the whole file).
 * @param bytes - File head; the data chunk header must fall within it.
 */
export function wavDurationSeconds(bytes: Uint8Array): number | null {
	if (bytes.length < 12) {
		return null;
	}
	let byteRate = 0;
	let offset = 12;
	while (offset + 8 <= bytes.length) {
		const id = String.fromCharCode(
			bytes[offset]!,
			bytes[offset + 1]!,
			bytes[offset + 2]!,
			bytes[offset + 3]!,
		);
		const size = readU32LE(bytes, offset + 4);
		if (id === 'fmt ' && offset + 24 <= bytes.length) {
			byteRate = readU32LE(bytes, offset + 16);
		} else if (id === 'data') {
			return byteRate ? size / byteRate : null;
		}
		offset += 8 + size + (size % 2);
	}
	return null;
}

function normalizeForDupe(text: string): string {
	return text.replace(/\s+/g, ' ').trim();
}

/**
 * Collapses adjacent segments that repeat the same text within
 * DUPE_GAP_SECONDS - residue a chunk boundary leaves behind - by extending
 * the first segment's end over its duplicates.
 */
export function dedupeAdjacentSegments(
	segments: TranscriptSegment[],
): TranscriptSegment[] {
	const kept: TranscriptSegment[] = [];
	for (const segment of segments) {
		const prev = kept[kept.length - 1];
		if (
			prev &&
			normalizeForDupe(prev.text) === normalizeForDupe(segment.text) &&
			segment.start - prev.end < DUPE_GAP_SECONDS
		) {
			prev.end = Math.max(prev.end, segment.end);
			continue;
		}
		kept.push(segment);
	}
	return kept;
}

/** Configuration for the optional local diarization step. */
export interface LocalDiarizerConfig {
	pythonPath: string;
	scriptPath: string;
	/** Expected speaker count; 0 lets the script estimate. */
	maxSpeakers: number;
}

/** Configuration for the local whisper.cpp provider. */
export interface LocalWhisperConfig {
	binaryPath: string;
	modelPath: string;
	extraArgs: string[];
	/** Time limit for one run, in milliseconds, from the user's setting. */
	processTimeoutMs: number;
	/** ffmpeg binary used to convert the original container to 16 kHz WAV. */
	ffmpegPath: string;
	diarizer?: LocalDiarizerConfig | undefined;
}

/**
 * Resolves Node builtins via Obsidian's desktop `require`, or null on
 * platforms (mobile) where they are unavailable.
 */
function loadNodeModules(): NodeModules | null {
	const req = (window as { require?: (id: string) => unknown }).require;
	if (typeof req !== 'function') {
		return null;
	}
	try {
		return {
			childProcess: req('child_process') as NodeModules['childProcess'],
			fs: req('fs') as NodeModules['fs'],
			os: req('os') as NodeModules['os'],
			path: req('path') as NodeModules['path'],
		};
	} catch {
		return null;
	}
}

/**
 * Maps whisper.cpp `-oj` JSON (offsets in milliseconds) to segments.
 * @param body - Parsed whisper.cpp JSON
 */
export function mapWhisperCppJson(body: unknown): WhisperResult {
	if (!isRecord(body)) {
		return { segments: [] };
	}
	const items = body.transcription;
	const language =
		typeof body.language === 'string' ? body.language : undefined;
	if (!Array.isArray(items)) {
		return { language, segments: [] };
	}
	const segments: TranscriptSegment[] = [];
	for (const entry of items) {
		if (!isRecord(entry)) {
			continue;
		}
		const offsets = isRecord(entry.offsets) ? entry.offsets : undefined;
		const text = typeof entry.text === 'string' ? entry.text.trim() : '';
		if (text === '') {
			continue;
		}
		const fromMs = offsets ? num(offsets.from) : 0;
		const toMs = offsets ? num(offsets.to, fromMs) : 0;
		segments.push({ start: fromMs / 1000, end: toMs / 1000, text });
	}
	return { language, segments };
}

/** Node's code for a child killed for outgrowing `maxBuffer`. */
const MAX_BUFFER_ERROR_CODE = 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER';

/**
 * Says which of the four ways a run can end badly this error is.
 *
 * Node reports a process it killed the same way it reports one that exited
 * with a status, so the error alone reads as "the binary failed" whether the
 * user pressed Cancel, the limit ran out, the output outgrew the buffer, or
 * the model path was wrong. Only the caller can tell them apart, because only
 * it holds the cancel and set the limit.
 *
 * The fourth is the binary's own to explain, and it does explain it: a model
 * it could not load or a flag it does not know is named on stderr, while the
 * error beside it says no more than that the exit was non-zero. Every ending
 * keeps the original as its `cause`, so the console still has what Node saw.
 * @param error - What execFile handed back
 * @param options - The run's options, holding its cancel
 * @param stderr - What the binary wrote to stderr, empty when it wrote nothing
 * @returns The error to reject with
 */
function describeRunFailure(
	error: Error,
	options: TranscribeOptions,
	stderr: string,
): Error {
	if (options.signal?.aborted) {
		return new Error('Local whisper.cpp run was cancelled.', {
			cause: error,
		});
	}
	const { killed, code } = error as { killed?: boolean; code?: string };
	// Ahead of the branch below, which reads the same `killed` marker: Node
	// sets it for a child it stopped for writing too much as well as for one
	// it stopped for running too long, and raising a time limit that was never
	// reached does nothing for the first.
	if (code === MAX_BUFFER_ERROR_CODE) {
		return new Error(
			'Local whisper.cpp produced more output than the plugin can read. ' +
				'Split the recording and transcribe it in parts.',
			{ cause: error },
		);
	}
	// Node's own marker for a process it killed, which here means the limit.
	if (killed) {
		return new Error(
			'Local whisper.cpp did not finish within the run timeout and was ' +
				'stopped. Raise Local run timeout in the settings, or use a ' +
				'smaller model.',
			{ cause: error },
		);
	}
	// Everything else is the binary's own failure, and the last line it wrote
	// is where it says which one. Node's message for it is "Command failed
	// with exit code 1", which sends a user with a mistyped model path looking
	// for a status code instead of the sentence naming the file.
	const detail = stderr.trim().split('\n').at(-1);
	if (!detail) {
		return error;
	}
	return new Error(`Local whisper.cpp failed: ${detail}`, { cause: error });
}

/** Removes a temp path, warning instead of throwing when it is still held. */
function removeTempFile(
	node: NodeModules,
	path: string,
	options?: { recursive?: boolean },
): void {
	try {
		node.fs.rmSync(path, { force: true, ...options });
	} catch (error) {
		// `force` answers a file that is not there and nothing else, and the
		// run can now end while the binary is still holding its input open:
		// Node kills the child on the cancel or the timeout, and the OS may
		// refuse to unlink a file with a live handle. A throw from here would
		// replace the reason the run ended, so the user would be told about a
		// temp path instead of about their own Cancel.
		console.warn(
			`${PLUGIN_LOG_PREFIX} Temporary whisper.cpp file could not be removed:`,
			path,
			error,
		);
	}
}

/**
 * Local whisper.cpp transcription provider (desktop only).
 */
export class LocalWhisperProvider implements TranscriptionProvider {
	readonly id = TRANSCRIPTION_PROVIDER_IDS.LOCAL_WHISPER;
	readonly label = 'Local whisper.cpp';
	readonly requiresNetwork = false;
	readonly capabilities: ProviderCapabilities = LOCAL_WHISPER_CAPABILITIES;
	private readonly node = loadNodeModules();

	constructor(private readonly config: LocalWhisperConfig) {}

	/** Whether the local binary path can be invoked on this platform. */
	isAvailable(): boolean {
		return this.node !== null;
	}

	async transcribe(
		payload: AudioPayload,
		options: TranscribeOptions,
	): Promise<WhisperResult> {
		const node = this.node;
		if (!node) {
			throw new Error(
				'Local transcription is only available in the desktop app.',
			);
		}
		const base = node.path.join(
			node.os.tmpdir(),
			`aar-whisper-${String(Date.now())}-${randomToken()}`,
		);
		const bytes = new Uint8Array(payload.data);
		const isWav =
			bytes.length > 12 &&
			bytes[0] === 0x52 && // 'R'
			bytes[1] === 0x49 && // 'I'
			bytes[2] === 0x46 && // 'F'
			bytes[3] === 0x46; // 'F'
		const wavPath = `${base}.wav`;
		if (isWav) {
			node.fs.writeFileSync(wavPath, bytes);
		} else {
			// The service sent the original container (webm/m4a/...): convert
			// with the desktop ffmpeg instead of the renderer's WebAudio
			// decode, which is memory-bounded and fails on long meetings.
			const rawPath = `${base}.in`;
			node.fs.writeFileSync(rawPath, bytes);
			try {
				await this.runFfmpeg(
					node,
					[
						'-nostdin', '-v', 'error', '-y',
						'-i', rawPath,
						'-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', wavPath,
					],
					'ffmpeg could not convert the recording',
					options,
				);
			} finally {
				removeTempFile(node, rawPath);
			}
		}

		// whisper.cpp shares Whisper's single prompt slot and its ~224-token
		// window, so it shares the policy that fills the slot.
		const promptValue = whisperPromptValue(options);

		// Split long recordings into independent CHUNK_SECONDS runs. The
		// segment muxer cuts the shared 16 kHz timeline at exact marks, so
		// part i starts at i * CHUNK_SECONDS and merging only shifts
		// timestamps back onto the whole-recording timeline - diarization
		// below still runs on the full WAV with global speaker numbering.
		const duration = this.readWavDuration(node, wavPath);
		const chunkDir = `${base}.chunks`;
		let inputs: string[] = [wavPath];
		let splitIntoChunks = false;
		if (duration !== null && duration > LOCAL_WHISPER_CHUNK_SECONDS) {
			splitIntoChunks = true;
			node.fs.mkdirSync(chunkDir);
			await this.runFfmpeg(
				node,
				[
					'-nostdin', '-v', 'error', '-y',
					'-i', wavPath,
					'-f', 'segment',
					'-segment_time', String(LOCAL_WHISPER_CHUNK_SECONDS),
					'-c', 'copy',
					node.path.join(chunkDir, 'chunk_%04d.wav'),
				],
				'ffmpeg could not split the recording for chunked transcription',
				options,
			);
			const chunks = node.fs
				.readdirSync(chunkDir)
				.filter((name) => /^chunk_\d+\.wav$/.test(name))
				.sort()
				.map((name) => node.path.join(chunkDir, name));
			if (chunks.length) {
				inputs = chunks;
			}
		}

		const partJsons: string[] = [];
		const result: WhisperResult = { segments: [] };
		try {
			let language: string | undefined;
			for (let i = 0; i < inputs.length; i++) {
				if (options.signal?.aborted) {
					throw new Error('Local whisper.cpp run was cancelled.');
				}
				const partBase = `${base}.part${String(i).padStart(4, '0')}`;
				partJsons.push(`${partBase}.json`);
				if (inputs.length > 1) {
					console.debug(
						`${PLUGIN_LOG_PREFIX} local whisper part ${i + 1}/${inputs.length}`,
					);
				}
				await this.runWhisper(
					node,
					inputs[i]!,
					partBase,
					promptValue,
					options,
				);
				const part = this.readWhisperJson(node, partBase);
				language ??= part.language;
				const offset =
					inputs.length > 1 ? i * LOCAL_WHISPER_CHUNK_SECONDS : 0;
				for (const segment of part.segments) {
					result.segments.push({
						start: segment.start + offset,
						end: segment.end + offset,
						text: segment.text,
					});
				}
			}
			result.language = language;
			result.segments = dedupeAdjacentSegments(result.segments);
			// Diarization is a best-effort enhancement: a failure must never
			// discard the transcript whisper already produced.
			if (options.diarize && this.config.diarizer && result.segments.length) {
				try {
					await this.applyDiarization(node, wavPath, result.segments, options);
				} catch (error) {
					console.warn(
						`${PLUGIN_LOG_PREFIX} local diarization failed:`,
						error,
					);
				}
			}
			return result;
		} finally {
			removeTempFile(node, wavPath);
			for (const path of partJsons) {
				removeTempFile(node, path);
			}
			if (splitIntoChunks) {
				removeTempFile(node, chunkDir, { recursive: true });
			}
		}
	}

	/**
	 * Runs whisper.cpp over one WAV part. `--max-context 0` keeps runs
	 * independent - the phrase-loop fix - and sits before extraArgs so a
	 * user-supplied flag can still override it. The decoder-fail thresholds
	 * follow for the same reason: on noise/silence or faint far-mic speech the
	 * default thresholds let the decoder lock onto the previous phrase and
	 * repeat it for minutes (a 9/7 meeting lost 8 minutes of real speech to a
	 * "PM분들의 노하우를 넣었다" loop). Verified on that recording: entropy
	 * 2.6 / logprob -0.8 / no-speech 0.5 recover the underlying speech
	 * instead of looping.
	 */
	private runWhisper(
		node: NodeModules,
		wavPath: string,
		partBase: string,
		promptValue: string | undefined,
		options: TranscribeOptions,
	): Promise<void> {
		const args = [
			'-m',
			this.config.modelPath,
			'-f',
			wavPath,
			'-oj',
			'-of',
			partBase,
			...(options.language ? ['-l', options.language] : []),
			...(promptValue ? ['--prompt', promptValue] : []),
			'--max-context',
			'0',
			'--entropy-thold',
			'2.6',
			'--logprob-thold',
			'-0.8',
			'--no-speech-thold',
			'0.5',
			...this.config.extraArgs,
		];
		return new Promise<void>((resolve, reject) => {
			node.childProcess.execFile(
				this.config.binaryPath,
				args,
				{
					// whisper.cpp streams the full transcript to stdout; raise the
					// buffer so a long recording is not killed at Node's 1 MB default.
					maxBuffer: LOCAL_WHISPER_MAX_BUFFER_BYTES,
					// Node kills the process on either of these, which is
					// what makes the dialog releasable. Without them a
					// binary that never returns left the promise pending
					// until Obsidian was restarted, and Cancel had nothing
					// to act on.
					timeout: this.config.processTimeoutMs,
					...(options.signal ? { signal: options.signal } : {}),
				},
				(error, _stdout, stderr) => {
					if (error) {
						reject(describeRunFailure(error, options, stderr));
					} else {
						resolve();
					}
				},
			);
		});
	}

	/** Reads and maps one whisper.cpp `-oj` output file. */
	private readWhisperJson(node: NodeModules, partBase: string): WhisperResult {
		let raw: string;
		try {
			raw = node.fs.readFileSync(`${partBase}.json`, 'utf8');
		} catch {
			throw new Error(
				'Local whisper.cpp did not produce an output file. Check the binary and model paths.',
			);
		}
		let parsed: unknown;
		try {
			parsed = JSON.parse(raw);
		} catch {
			throw new Error('Local whisper.cpp produced invalid JSON output.');
		}
		return mapWhisperCppJson(parsed);
	}

	/**
	 * Duration of the prepared WAV, or null when the header is not parseable
	 * (chunking is then skipped and the whole file runs as before).
	 */
	private readWavDuration(node: NodeModules, wavPath: string): number | null {
		let fd: number;
		try {
			fd = node.fs.openSync(wavPath, 'r');
		} catch {
			return null;
		}
		try {
			// The data chunk header follows any metadata chunks; 64 KB covers
			// every header ffmpeg writes without loading the whole recording.
			const head = new Uint8Array(65536);
			const read = node.fs.readSync(fd, head, 0, head.length, 0);
			return wavDurationSeconds(head.subarray(0, read));
		} catch {
			return null;
		} finally {
			node.fs.closeSync(fd);
		}
	}

	/**
	 * Runs ffmpeg with a full argument list (container conversion or chunk
	 * segmentation), bounded and cancelled like the whisper runs.
	 */
	private runFfmpeg(
		node: NodeModules,
		args: string[],
		failure: string,
		options: TranscribeOptions,
	): Promise<void> {
		return new Promise<void>((resolve, reject) => {
			node.childProcess.execFile(
				this.config.ffmpegPath || 'ffmpeg',
				args,
				{
					maxBuffer: LOCAL_WHISPER_MAX_BUFFER_BYTES,
					timeout: this.config.processTimeoutMs,
					...(options.signal ? { signal: options.signal } : {}),
				},
				(error, _stdout, stderr) => {
					if (error) {
						reject(
							new Error(
								`${failure}: ${String(describeRunFailure(error, options, stderr).message)}`,
							),
						);
					} else {
						resolve();
					}
				},
			);
		});
	}

	/**
	 * Runs the external diarization script over the transcribed WAV and
	 * attaches the speaker labels it reports to the matching segments.
	 * The script contract: `--wav <path> --segments <json> --max-speakers <n>`
	 * printing `{"speakers": N, "assignments": [{"index", "speaker"}]}`.
	 */
	private async applyDiarization(
		node: NodeModules,
		wavPath: string,
		segments: TranscriptSegment[],
		options: TranscribeOptions,
	): Promise<void> {
		const diarizer = this.config.diarizer;
		if (!diarizer) {
			return;
		}
		const segPath = `${wavPath}.segments.json`;
		node.fs.writeFileSync(
			segPath,
			JSON.stringify(
				segments.map((s) => ({ start: s.start, end: s.end, text: s.text })),
			),
		);
		try {
			const stdout = await new Promise<string>((resolve, reject) => {
				node.childProcess.execFile(
					diarizer.pythonPath,
					[
						diarizer.scriptPath,
						'--wav',
						wavPath,
						'--segments',
						segPath,
						'--max-speakers',
						String(diarizer.maxSpeakers),
					],
					{
						maxBuffer: LOCAL_WHISPER_MAX_BUFFER_BYTES,
						timeout: this.config.processTimeoutMs,
						...(options.signal ? { signal: options.signal } : {}),
					},
					(error, out) => {
						if (error) {
							reject(error);
						} else {
							resolve(out);
						}
					},
				);
			});
			const parsed = JSON.parse(stdout) as {
				assignments?: Array<{ index?: number; speaker?: string }>;
			};
			if (!Array.isArray(parsed.assignments)) {
				throw new Error('diarization script returned no assignments');
			}
			for (const entry of parsed.assignments) {
				const i =
					typeof entry.index === 'number' ? entry.index : Number.NaN;
				if (
					Number.isInteger(i) &&
					i >= 0 &&
					i < segments.length &&
					entry.speaker
				) {
					segments[i]!.speaker = entry.speaker;
				}
			}
		} finally {
			removeTempFile(node, segPath);
		}
	}
}
