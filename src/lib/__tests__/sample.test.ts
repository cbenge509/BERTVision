import {describe, it, expect} from 'vitest';

describe('Sample test suite', () => {
  it('should pass a basic assertion', () => {
    expect(1 + 1).toBe(2);
  });

  it('should work with strings', () => {
    expect('BERTVision').toContain('BERT');
  });
});
