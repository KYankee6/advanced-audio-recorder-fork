/**
 * Transcription via a local whisper.cpp binary. Desktop only - it shells
 * out through Node's child_process, which is unavailable in the mobile
 * app. The service hands this provider a single decoded WAV per request (it
 * declares no upload limit), which is written to a temp file, split into
 * independent CHUNK_SECONDS parts when long (whole-file runs loop on a
 * repeated phrase), transcribed to JSON, merged on the original timeline,
 * then all temp files are removed.
 * @module transcription/providers/LocalWhisperProvider
 */

import {
	LOCAL_WHISPER_MAX_BUFFER_BYTES,
	TRANSCRIPTION_PROVIDER_IDS,
} from '../../constants';
import {
	DICTIONARY_JOIN_SEPARATOR,
	termsWithinWhisperPrompt,
} from '../dictionaryBias';
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

/** Minimal Node surface used by the local provider. */
interface NodeModules {
	childProcess: {
		execFile: (
			file: string,
			args: string[],
			options: { maxBuffer: number },
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
			(bytes[offset + 3]! << 24)) >>>
		0
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
						'-nostdin',
						'-v',
						'error',
						'-y',
						'-i',
						rawPath,
						'-ar',
						'16000',
						'-ac',
						'1',
						'-c:a',
						'pcm_s16le',
						wavPath,
					],
					'ffmpeg could not convert the recording',
				);
			} finally {
				node.fs.rmSync(rawPath, { force: true });
			}
		}

		// The advanced second pass supplies a full bias sentence that already
		// folds the relevant terms in; it takes the --prompt slot over the plain
		// dictionary join. Otherwise, whisper.cpp shares Whisper's ~224-token
		// prompt window, so bound the dictionary the same way the service does
		// before building the flag.
		const biasPrompt = options.biasPrompt?.trim();
		const promptTerms =
			!biasPrompt && options.dictionary?.length
				? termsWithinWhisperPrompt(options.dictionary)
				: [];
		const promptValue =
			biasPrompt ??
			(promptTerms.length
				? promptTerms.join(DICTIONARY_JOIN_SEPARATOR)
				: undefined);

		// Split long recordings into independent CHUNK_SECONDS runs. The
		// segment muxer cuts the shared 16 kHz timeline at exact marks, so
		// part i starts at i * CHUNK_SECONDS and merging only shifts
		// timestamps back onto the whole-recording timeline - diarization
		// below still runs on the full WAV with global speaker numbering.
		const duration = this.readWavDuration(node, wavPath);
		const chunkDir = `${base}.chunks`;
		let inputs: string[] = [wavPath];
		if (duration !== null && duration > LOCAL_WHISPER_CHUNK_SECONDS) {
			node.fs.mkdirSync(chunkDir);
			await this.runFfmpeg(
				node,
				[
					'-nostdin',
					'-v',
					'error',
					'-y',
					'-i',
					wavPath,
					'-f',
					'segment',
					'-segment_time',
					String(LOCAL_WHISPER_CHUNK_SECONDS),
					'-c',
					'copy',
					node.path.join(chunkDir, 'chunk_%04d.wav'),
				],
				'ffmpeg could not split the recording for chunked transcription',
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
				const partBase = `${base}.part${String(i).padStart(4, '0')}`;
				partJsons.push(`${partBase}.json`);
				if (inputs.length > 1) {
					console.debug(
						`[advanced-audio-recorder] local whisper part ${i + 1}/${inputs.length}`,
					);
				}
				await this.runWhisper(
					node,
					inputs[i]!,
					partBase,
					promptValue,
					options.language,
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
			if (
				options.diarize &&
				this.config.diarizer &&
				result.segments.length
			) {
				try {
					await this.applyDiarization(node, wavPath, result.segments);
				} catch (error) {
					console.warn(
						'[advanced-audio-recorder] local diarization failed:',
						error,
					);
				}
			}
			return result;
		} finally {
			node.fs.rmSync(wavPath, { force: true });
			for (const path of partJsons) {
				node.fs.rmSync(path, { force: true });
			}
			node.fs.rmSync(chunkDir, { force: true, recursive: true });
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
		language: string | undefined,
	): Promise<void> {
		const args = [
			'-m',
			this.config.modelPath,
			'-f',
			wavPath,
			'-oj',
			'-of',
			partBase,
			...(language ? ['-l', language] : []),
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
				// whisper.cpp streams the full transcript to stdout; raise the
				// buffer so a long recording is not killed at Node's 1 MB default.
				{ maxBuffer: LOCAL_WHISPER_MAX_BUFFER_BYTES },
				(error) => {
					if (error) {
						reject(error);
					} else {
						resolve();
					}
				},
			);
		});
	}

	/** Reads and maps one whisper.cpp `-oj` output file. */
	private readWhisperJson(
		node: NodeModules,
		partBase: string,
	): WhisperResult {
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
	 * segmentation).
	 */
	private runFfmpeg(
		node: NodeModules,
		args: string[],
		failure: string,
	): Promise<void> {
		return new Promise<void>((resolve, reject) => {
			node.childProcess.execFile(
				this.config.ffmpegPath || 'ffmpeg',
				args,
				{ maxBuffer: LOCAL_WHISPER_MAX_BUFFER_BYTES },
				(error) => {
					if (error) {
						reject(
							new Error(`${failure}: ${String(error.message)}`),
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
	): Promise<void> {
		const diarizer = this.config.diarizer;
		if (!diarizer) {
			return;
		}
		const segPath = `${wavPath}.segments.json`;
		node.fs.writeFileSync(
			segPath,
			JSON.stringify(
				segments.map((s) => ({
					start: s.start,
					end: s.end,
					text: s.text,
				})),
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
					{ maxBuffer: LOCAL_WHISPER_MAX_BUFFER_BYTES },
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
			node.fs.rmSync(segPath, { force: true });
		}
	}
}
