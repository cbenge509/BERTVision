import {describe, it, expect} from 'vitest';
import {
  normalizeText,
  fuzzyMatch,
  extractContext,
  search,
  type SearchItem,
} from '../search';

describe('search utilities', () => {
  describe('normalizeText', () => {
    it('should convert text to lowercase', () => {
      expect(normalizeText('Hello World')).toBe('hello world');
    });

    it('should trim whitespace', () => {
      expect(normalizeText('  hello  ')).toBe('hello');
    });

    it('should collapse multiple spaces', () => {
      expect(normalizeText('hello    world')).toBe('hello world');
    });

    it('should handle empty string', () => {
      expect(normalizeText('')).toBe('');
    });
  });

  describe('fuzzyMatch', () => {
    it('should return 0 for empty query', () => {
      expect(fuzzyMatch('', 'hello world')).toBe(0);
    });

    it('should return high score for exact match', () => {
      const score = fuzzyMatch('hello', 'hello world');
      expect(score).toBeGreaterThanOrEqual(80);
    });

    it('should return higher score for word boundary match', () => {
      const boundaryScore = fuzzyMatch('hello', 'hello world');
      const substringScore = fuzzyMatch('ello', 'hello world');
      expect(boundaryScore).toBeGreaterThan(substringScore);
    });

    it('should match partial words', () => {
      const score = fuzzyMatch('bert', 'BERTVision is a model');
      expect(score).toBeGreaterThan(0);
    });

    it('should return 0 when no match', () => {
      expect(fuzzyMatch('xyz', 'hello world')).toBe(0);
    });

    it('should handle multiple query words', () => {
      const score = fuzzyMatch('hello world', 'hello beautiful world');
      expect(score).toBeGreaterThan(0);
    });

    it('should give proximity bonus for close words', () => {
      const closeScore = fuzzyMatch('hello world', 'hello world');
      const farScore = fuzzyMatch(
        'hello world',
        'hello this is a very long text that separates the words world'
      );
      expect(closeScore).toBeGreaterThan(farScore);
    });
  });

  describe('extractContext', () => {
    const longText =
      'This is a very long text that contains the word BERTVision somewhere in the middle of the content and continues for a while after that.';

    it('should extract context around match', () => {
      const context = extractContext(longText, 'BERTVision');
      expect(context).toContain('BERTVision');
      expect(context.length).toBeLessThan(longText.length);
    });

    it('should add ellipsis when truncating', () => {
      const context = extractContext(longText, 'BERTVision');
      expect(context).toContain('...');
    });

    it('should return beginning when no match', () => {
      const context = extractContext(longText, 'xyz', 50);
      expect(context.startsWith('This is')).toBe(true);
    });

    it('should handle short text', () => {
      const shortText = 'hello world';
      const context = extractContext(shortText, 'hello');
      expect(context).toBe('hello world');
    });
  });

  describe('search', () => {
    const items: SearchItem[] = [
      {
        id: 'intro',
        title: 'Introduction',
        content: 'BERTVision is a parameter-efficient approach for NLP tasks.',
        href: '#intro',
        type: 'section',
      },
      {
        id: 'results',
        title: 'Results',
        content: 'The model achieves state-of-the-art performance on GLUE benchmark.',
        href: '#results',
        type: 'section',
      },
      {
        id: 'data',
        title: 'Data and Methodology',
        content: 'We use SQuAD 2.0 and GLUE datasets for evaluation.',
        href: '#data',
        type: 'section',
      },
    ];

    it('should return empty array for empty query', () => {
      expect(search(items, '')).toEqual([]);
      expect(search(items, '   ')).toEqual([]);
    });

    it('should find items by title', () => {
      const results = search(items, 'Introduction');
      expect(results.length).toBeGreaterThan(0);
      expect(results[0]?.item.id).toBe('intro');
    });

    it('should find items by content', () => {
      const results = search(items, 'GLUE benchmark');
      expect(results.length).toBeGreaterThan(0);
    });

    it('should weight title matches higher', () => {
      const results = search(items, 'Results');
      expect(results[0]?.item.id).toBe('results');
    });

    it('should sort by score descending', () => {
      const results = search(items, 'data');
      for (let i = 1; i < results.length; i++) {
        const prev = results[i - 1];
        const curr = results[i];
        if (prev && curr) {
          expect(prev.score).toBeGreaterThanOrEqual(curr.score);
        }
      }
    });

    it('should respect limit parameter', () => {
      const results = search(items, 'the', 1);
      expect(results.length).toBeLessThanOrEqual(1);
    });

    it('should include context in results', () => {
      const results = search(items, 'BERTVision');
      expect(results[0]?.context).toBeDefined();
      expect(results[0]?.context.length).toBeGreaterThan(0);
    });
  });
});
