/**
 * The provider fields no declarative control type covers: the local
 * whisper.cpp file paths.
 * @module settings/sections/transcriptionEngineSection
 */

import {
	LOCAL_WHISPER_MODEL_NAMES,
	LOCAL_WHISPER_MODELS_DOC_URL,
	TRANSCRIPTION_PROVIDER_IDS,
} from '../../constants';
import { addText, type SettingsSectionContext } from '../settingControls';
import { isProviderAvailableOnPlatform } from '../../transcription/providers/capabilities';

/** Local whisper.cpp engine fields (binary path, model path, extra args). */
export function renderLocalWhisperSettings(ctx: SettingsSectionContext): void {
	const s = ctx.settings;
	// A stored local-whisper selection synced to a platform that cannot
	// run it (mobile) keeps its fields visible but blocked, so the user
	// sees why transcription will not run instead of editable settings
	// that quietly do nothing.
	const unavailable = !isProviderAvailableOnPlatform(
		TRANSCRIPTION_PROVIDER_IDS.LOCAL_WHISPER,
	);
	const unavailableHint = unavailable
		? ' Local whisper.cpp is not available on this device; pick a cloud engine to transcribe here.'
		: '';
	addText(ctx, {
		name: 'whisper.cpp binary path',
		desc: `Absolute path to the whisper.cpp executable.${unavailableHint}`,
		get: () => s.localWhisperBinaryPath,
		set: (v) => (s.localWhisperBinaryPath = v),
		disabled: unavailable,
	});
	addText(ctx, {
		name: 'Model path',
		desc: `Absolute path to a GGML model file (.bin). Download one of: ${LOCAL_WHISPER_MODEL_NAMES.join(', ')} (names ending in .en are English-only).`,
		helpLink: {
			label: 'Download whisper.cpp models',
			url: LOCAL_WHISPER_MODELS_DOC_URL,
		},
		get: () => s.localWhisperModelPath,
		set: (v) => (s.localWhisperModelPath = v),
		disabled: unavailable,
	});
	addText(ctx, {
		name: 'Extra arguments',
		desc: 'Optional extra CLI arguments, space-separated.',
		get: () => s.localWhisperExtraArgs,
		set: (v) => (s.localWhisperExtraArgs = v),
		disabled: unavailable,
	});
	addText(ctx, {
		name: 'ffmpeg path',
		desc: 'ffmpeg binary that converts the original recording (webm/m4a) to WAV before local transcription. Long meetings no longer go through the memory-bounded in-app decoder.',
		get: () => s.localFfmpegPath,
		set: (v) => (s.localFfmpegPath = v),
		disabled: unavailable,
	});
	// Local diarization (speaker labeling) runs our python script over the
	// same WAV whisper just transcribed. Empty python path turns it off.
	addText(ctx, {
		name: 'Diarization python path',
		desc: 'Python interpreter for local speaker diarization. Leave empty to disable diarization.',
		get: () => s.localDiarizerPythonPath,
		set: (v) => (s.localDiarizerPythonPath = v),
		disabled: unavailable,
	});
	addText(ctx, {
		name: 'Diarization script path',
		desc: 'Path to the diarization script (takes the WAV and whisper segments, prints speaker assignments as JSON).',
		get: () => s.localDiarizerScriptPath,
		set: (v) => (s.localDiarizerScriptPath = v),
		disabled: unavailable,
	});
	addText(ctx, {
		name: 'Expected speakers',
		desc: 'Speaker count for diarization. 0 = estimate automatically (set the count when you know it).',
		get: () => String(s.localDiarizerMaxSpeakers),
		set: (v) => {
			const n = Number.parseInt(v, 10);
			s.localDiarizerMaxSpeakers = Number.isFinite(n) && n >= 0 ? n : 0;
		},
		disabled: unavailable,
	});
}
