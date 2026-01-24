/**
 * Search utilities for the SearchModal component.
 *
 * Provides fuzzy search functionality for page content.
 */

/**
 * A searchable content item.
 */
export interface SearchItem {
  /** Unique identifier */
  id: string;
  /** Display title */
  title: string;
  /** Searchable content text */
  content: string;
  /** Section anchor (e.g., #introduction) */
  href: string;
  /** Content type for display */
  type: 'section' | 'heading' | 'content';
}

/**
 * Search result with matching context.
 */
export interface SearchResult {
  /** The matched item */
  item: SearchItem;
  /** Relevance score (higher is better) */
  score: number;
  /** Context snippet around the match */
  context: string;
}

/**
 * Normalizes a string for search comparison.
 * Converts to lowercase and removes extra whitespace.
 */
export function normalizeText(text: string): string {
  return text.toLowerCase().trim().replace(/\s+/g, ' ');
}

/**
 * Simple fuzzy matching - checks if query words appear in text.
 * Returns a score based on match quality.
 */
export function fuzzyMatch(query: string, text: string): number {
  const normalizedQuery = normalizeText(query);
  const normalizedText = normalizeText(text);

  if (!normalizedQuery) return 0;

  // Exact match gets highest score
  if (normalizedText.includes(normalizedQuery)) {
    // Bonus for match at word boundary
    const wordBoundaryPattern = new RegExp(`\\b${escapeRegex(normalizedQuery)}`, 'i');
    if (wordBoundaryPattern.test(normalizedText)) {
      return 100;
    }
    return 80;
  }

  // Check if all query words appear in text
  const queryWords = normalizedQuery.split(' ').filter(w => w.length > 0);
  const matchedWords = queryWords.filter(word => normalizedText.includes(word));

  if (matchedWords.length === 0) return 0;

  // Score based on percentage of query words matched
  const wordMatchScore = (matchedWords.length / queryWords.length) * 60;

  // Bonus for words appearing close together
  let proximityBonus = 0;
  if (matchedWords.length > 1) {
    const firstWord = matchedWords[0];
    const lastWord = matchedWords[matchedWords.length - 1];
    if (firstWord !== undefined && lastWord !== undefined) {
      const firstWordIndex = normalizedText.indexOf(firstWord);
      const lastWordIndex = normalizedText.indexOf(lastWord);
      const distance = Math.abs(lastWordIndex - firstWordIndex);
      if (distance < 50) {
        proximityBonus = 20 * (1 - distance / 50);
      }
    }
  }

  return wordMatchScore + proximityBonus;
}

/**
 * Escapes special regex characters in a string.
 */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Extracts a context snippet around a match.
 */
export function extractContext(text: string, query: string, maxLength = 120): string {
  const normalizedQuery = normalizeText(query);
  const normalizedText = normalizeText(text);

  // Find the match position
  const queryWords = normalizedQuery.split(' ').filter(w => w.length > 0);
  let matchStart = -1;

  for (const word of queryWords) {
    const index = normalizedText.indexOf(word);
    if (index !== -1 && (matchStart === -1 || index < matchStart)) {
      matchStart = index;
    }
  }

  if (matchStart === -1) {
    // No match found, return beginning of text
    return text.slice(0, maxLength) + (text.length > maxLength ? '...' : '');
  }

  // Calculate context window
  const contextPadding = Math.floor((maxLength - normalizedQuery.length) / 2);
  let start = Math.max(0, matchStart - contextPadding);
  let end = Math.min(text.length, matchStart + normalizedQuery.length + contextPadding);

  // Adjust to word boundaries
  if (start > 0) {
    const spaceIndex = text.indexOf(' ', start);
    if (spaceIndex !== -1 && spaceIndex < start + 20) {
      start = spaceIndex + 1;
    }
  }

  if (end < text.length) {
    const spaceIndex = text.lastIndexOf(' ', end);
    if (spaceIndex !== -1 && spaceIndex > end - 20) {
      end = spaceIndex;
    }
  }

  let context = text.slice(start, end);

  // Add ellipsis
  if (start > 0) context = '...' + context;
  if (end < text.length) context = context + '...';

  return context;
}

/**
 * Searches through items and returns ranked results.
 */
export function search(items: SearchItem[], query: string, limit = 10): SearchResult[] {
  if (!query.trim()) return [];

  const results: SearchResult[] = [];

  for (const item of items) {
    // Search in title and content
    const titleScore = fuzzyMatch(query, item.title) * 1.5; // Title matches weighted higher
    const contentScore = fuzzyMatch(query, item.content);
    const score = Math.max(titleScore, contentScore);

    if (score > 0) {
      const searchTarget = titleScore > contentScore ? item.title : item.content;
      results.push({
        item,
        score,
        context: extractContext(searchTarget, query),
      });
    }
  }

  // Sort by score descending
  results.sort((a, b) => b.score - a.score);

  return results.slice(0, limit);
}

/**
 * Builds a search index from the page DOM.
 * Should be called after the page loads.
 */
export function buildSearchIndex(): SearchItem[] {
  const items: SearchItem[] = [];
  const sections = document.querySelectorAll('section[id]');

  sections.forEach(section => {
    const id = section.id;
    const heading = section.querySelector('h1, h2, h3');
    const title = heading?.textContent?.trim() ?? id;

    // Get text content excluding scripts and styles
    const clone = section.cloneNode(true) as Element;
    clone.querySelectorAll('script, style').forEach(el => el.remove());
    const content = clone.textContent?.trim() ?? '';

    items.push({
      id,
      title,
      content,
      href: `#${id}`,
      type: 'section',
    });
  });

  return items;
}
