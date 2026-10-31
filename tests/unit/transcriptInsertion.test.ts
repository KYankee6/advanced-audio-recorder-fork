/**
 * Tests the note-insertion half of transcriptOutput. Insertion targets the note
 * the timecode links were generated against - not whatever pane happens to be
 * active when an async run finishes - so the view lookup is what keeps a long
 * transcription from dumping its output into an unrelated file the user
 * switched to mid-run. When the note exists but no editable view holds it
 * (closed tab, reading mode), the block is appended to the file instead: a
 * run that finished while the user was elsewhere must still land. False is
 * reserved for a note that does not exist at all.
 * @module tests/unit/transcriptInsertion.test
 */

import {
	insertTranscriptFileLink,
	insertTranscriptIntoNote,
	writeTranscriptFile,
} from 'src/transcription/transcriptOutput';
import { MarkdownView } from 'obsidian';
import type { App, TFile } from 'obsidian';
import { at, defined } from '../helpers/assertions';
import { createMockApp } from '../helpers/createApp';

/** A Markdown leaf whose editor records the ranges written into it. */
function makeLeaf(
	notePath: string | null,
	options: { throws?: boolean } = {},
): {
	leaf: { view: MarkdownView };
	writes: [string, unknown][];
} {
	const writes: [string, unknown][] = [];
	// Built from the prototype rather than the constructor: the production
	// lookup filters leaves with `instanceof MarkdownView`, and Obsidian's own
	// constructor signature needs a real leaf we have no use for here.
	const view = Object.create(MarkdownView.prototype) as MarkdownView;
	Object.defineProperty(view, 'file', {
		value: notePath === null ? null : { path: notePath },
	});
	Object.defineProperty(view, 'editor', {
		get: () => ({
			getCursor: () => ({ line: 3, ch: 0 }),
			replaceRange: (text: string, cursor: unknown) => {
				if (options.throws) {
					throw new Error('editor is read-only');
				}
				writes.push([text, cursor]);
			},
		}),
	});
	return { leaf: { view }, writes };
}

/** An App exposing the given Markdown leaves. */
function makeApp(
	leaves: { view: MarkdownView }[],
	generateMarkdownLink = jest.fn().mockReturnValue('[[audio.srt]]'),
): App {
	return createMockApp({
		workspace: {
			getLeavesOfType: jest.fn((type: string) =>
				type === 'markdown' ? leaves : [],
			),
		},
		fileManager: { generateMarkdownLink },
	}).app;
}

describe('insertTranscriptIntoNote', () => {
	it('writes at the cursor of the note the links were generated against', () => {
		const target = makeLeaf('Notes/meeting.md');
		const other = makeLeaf('Notes/unrelated.md');
		const app = makeApp([other.leaf, target.leaf]);

		expect(
			insertTranscriptIntoNote(app, 'Notes/meeting.md', 'body', ''),
		).toBe(true);

		expect(at(target.writes, 0)).toEqual([
			'\n\nbody\n',
			{ line: 3, ch: 0 },
		]);
		// The pane the user switched to must be left alone.
		expect(other.writes).toEqual([]);
	});

	it('puts the heading above the body when one is given', () => {
		const target = makeLeaf('Notes/meeting.md');

		insertTranscriptIntoNote(
			makeApp([target.leaf]),
			'Notes/meeting.md',
			'body',
			'## Transcript',
		);

		expect(at(at(target.writes, 0), 0)).toBe('\n\n## Transcript\n\nbody\n');
	});

	it('appends to the note file when the target note is not open', () => {
		const other = makeLeaf('Notes/unrelated.md');
		const process = jest.fn(
			(_file: TFile, fn: (data: string) => string) => {
				expect(fn('existing')).toBe('existing\n\nbody\n');
				return Promise.resolve('existing\n\nbody\n');
			},
		);
		const app = createMockApp({
			workspace: {
				getLeavesOfType: jest.fn((type: string) =>
					type === 'markdown' ? [other.leaf] : [],
				),
			},
			vault: {
				getFileByPath: jest.fn((path: string) =>
					path === 'Notes/meeting.md' ? { path } : null,
				),
				process,
			},
		}).app;

		// A run that finished while the user was elsewhere still lands: the
		// block goes to the end of the note file instead of being dropped.
		expect(
			insertTranscriptIntoNote(app, 'Notes/meeting.md', 'body', ''),
		).toBe(true);
		expect(process).toHaveBeenCalled();
		// The pane the user switched to must be left alone.
		expect(other.writes).toEqual([]);
	});

	it('reports not-inserted when the note is neither open nor on disk', () => {
		const app = makeApp([]);

		expect(
			insertTranscriptIntoNote(app, 'Notes/meeting.md', 'body', ''),
		).toBe(false);
	});

	it('reports not-inserted for an empty note path', () => {
		const target = makeLeaf('Notes/meeting.md');

		// A run with no host note must not fall through to "the first open
		// Markdown leaf" - that is exactly the wrong-file bug.
		expect(
			insertTranscriptIntoNote(makeApp([target.leaf]), '', 'body', ''),
		).toBe(false);
		expect(target.writes).toEqual([]);
	});

	it('ignores a leaf whose view has no file', () => {
		const empty = makeLeaf(null);

		expect(
			insertTranscriptIntoNote(
				makeApp([empty.leaf]),
				'Notes/meeting.md',
				'body',
				'',
			),
		).toBe(false);
	});

	it('appends to the file when the editor refuses', () => {
		const target = makeLeaf('Notes/meeting.md', { throws: true });
		const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
		const process = jest.fn().mockResolvedValue('');
		const app = createMockApp({
			workspace: {
				getLeavesOfType: jest.fn((type: string) =>
					type === 'markdown' ? [target.leaf] : [],
				),
			},
			vault: {
				getFileByPath: jest.fn(() => ({ path: 'Notes/meeting.md' })),
				process,
			},
		}).app;

		// A note in reading mode: the editor refuses, so the block lands at
		// the end of the file instead of being dropped with a warning.
		expect(
			insertTranscriptIntoNote(app, 'Notes/meeting.md', 'body', ''),
		).toBe(true);
		expect(process).toHaveBeenCalled();
		expect(warn).toHaveBeenCalled();
	});
});

describe('insertTranscriptFileLink', () => {
	const transcriptFile = { path: 'Recordings/audio.srt' } as TFile;

	it('inserts a link built for the vault link format', () => {
		const target = makeLeaf('Notes/meeting.md');
		const generateMarkdownLink = jest
			.fn()
			.mockReturnValue('[[Recordings/audio.srt]]');
		const app = makeApp([target.leaf], generateMarkdownLink);

		expect(
			insertTranscriptFileLink(
				app,
				'Notes/meeting.md',
				transcriptFile,
				'',
			),
		).toBe(true);

		// Built through Obsidian rather than hand-formatted, so a wikilink vault
		// gets a wikilink and a Markdown-link vault gets a Markdown link.
		expect(generateMarkdownLink).toHaveBeenCalledWith(
			transcriptFile,
			'Notes/meeting.md',
		);
		expect(at(at(target.writes, 0), 0)).toBe(
			'\n\n[[Recordings/audio.srt]]\n',
		);
	});

	it('places the link under the heading when one is given', () => {
		const target = makeLeaf('Notes/meeting.md');

		insertTranscriptFileLink(
			makeApp([target.leaf]),
			'Notes/meeting.md',
			transcriptFile,
			'## Transcript',
		);

		expect(at(at(target.writes, 0), 0)).toBe(
			'\n\n## Transcript\n\n[[audio.srt]]\n',
		);
	});

	it('appends the link to the file when the note is not open', () => {
		const process = jest.fn().mockResolvedValue('');
		const app = createMockApp({
			vault: {
				getFileByPath: jest.fn(() => ({ path: 'Notes/meeting.md' })),
				process,
			},
		}).app;

		expect(
			insertTranscriptFileLink(
				app,
				'Notes/meeting.md',
				transcriptFile,
				'',
			),
		).toBe(true);
		expect(process).toHaveBeenCalled();
	});

	it('reports not-inserted when the note is neither open nor on disk', () => {
		expect(
			insertTranscriptFileLink(
				makeApp([]),
				'Notes/meeting.md',
				transcriptFile,
				'',
			),
		).toBe(false);
	});
});

describe('writeTranscriptFile', () => {
	it('writes the sidecar next to the audio in the requested format', async () => {
		const create = jest.fn().mockResolvedValue({ path: 'out' });
		const app = createMockApp({
			vault: {
				create,
				getFileByPath: jest.fn().mockReturnValue(null),
				adapter: { exists: jest.fn().mockResolvedValue(false) },
			},
		}).app;

		await writeTranscriptFile(
			app,
			{ path: 'Recordings/audio.webm' } as TFile,
			{
				language: 'en',
				speakers: [],
				segments: [{ start: 0, end: 1, text: 'hi' }],
			},
			'txt',
		);

		const [path, body] = at(create.mock.calls, 0);
		expect(path).toBe('Recordings/audio.txt');
		expect(defined(body)).toContain('hi');
	});

	it('does not overwrite an existing sidecar', async () => {
		const create = jest.fn().mockResolvedValue({ path: 'out' });
		const existing = new Set(['Recordings/audio.txt']);
		const app = createMockApp({
			vault: {
				create,
				getFileByPath: jest.fn((path: string) =>
					existing.has(path) ? { path } : null,
				),
				adapter: {
					exists: jest.fn((path: string) =>
						Promise.resolve(existing.has(path)),
					),
				},
			},
		}).app;

		await writeTranscriptFile(
			app,
			{ path: 'Recordings/audio.webm' } as TFile,
			{ language: 'en', speakers: [], segments: [] },
			'txt',
		);

		// Re-transcribing a file must not silently replace the previous
		// transcript.
		expect(at(at(create.mock.calls, 0), 0)).not.toBe(
			'Recordings/audio.txt',
		);
	});
});
