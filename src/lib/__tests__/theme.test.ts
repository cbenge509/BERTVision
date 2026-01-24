import {describe, it, expect, beforeEach, afterEach, vi} from 'vitest';
import {
  getSystemTheme,
  getStoredTheme,
  setStoredTheme,
  resolveTheme,
  applyTheme,
  getAppliedTheme,
  toggleTheme,
  setTheme,
  THEME_STORAGE_KEY,
  DARK_CLASS,
} from '../theme';

describe('theme', () => {
  // Mock localStorage
  const localStorageMock = (() => {
    let store: Record<string, string> = {};
    return {
      getItem: (key: string) => store[key] ?? null,
      setItem: (key: string, value: string) => {
        store[key] = value;
      },
      removeItem: (key: string) => {
        store = Object.fromEntries(
          Object.entries(store).filter(([k]) => k !== key)
        );
      },
      clear: () => {
        store = {};
      },
    };
  })();

  // Mock matchMedia
  let mockMatchMediaMatches = false;
  const mockMatchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: query === '(prefers-color-scheme: dark)' && mockMatchMediaMatches,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));

  beforeEach(() => {
    // Setup mocks
    Object.defineProperty(globalThis, 'localStorage', {
      value: localStorageMock,
      writable: true,
    });
    Object.defineProperty(globalThis, 'window', {
      value: {matchMedia: mockMatchMedia},
      writable: true,
    });
    Object.defineProperty(globalThis, 'document', {
      value: {
        documentElement: {
          classList: {
            _classes: new Set<string>(),
            add(cls: string) {
              this._classes.add(cls);
            },
            remove(cls: string) {
              this._classes.delete(cls);
            },
            contains(cls: string) {
              return this._classes.has(cls);
            },
          },
        },
      },
      writable: true,
    });

    // Reset state
    localStorageMock.clear();
    mockMatchMediaMatches = false;
    document.documentElement.classList.remove(DARK_CLASS);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('getSystemTheme', () => {
    it('returns "dark" when system prefers dark mode', () => {
      mockMatchMediaMatches = true;
      expect(getSystemTheme()).toBe('dark');
    });

    it('returns "light" when system prefers light mode', () => {
      mockMatchMediaMatches = false;
      expect(getSystemTheme()).toBe('light');
    });
  });

  describe('getStoredTheme', () => {
    it('returns null when no theme is stored', () => {
      expect(getStoredTheme()).toBeNull();
    });

    it('returns stored theme when valid', () => {
      localStorage.setItem(THEME_STORAGE_KEY, 'dark');
      expect(getStoredTheme()).toBe('dark');

      localStorage.setItem(THEME_STORAGE_KEY, 'light');
      expect(getStoredTheme()).toBe('light');

      localStorage.setItem(THEME_STORAGE_KEY, 'system');
      expect(getStoredTheme()).toBe('system');
    });

    it('returns null for invalid stored values', () => {
      localStorage.setItem(THEME_STORAGE_KEY, 'invalid');
      expect(getStoredTheme()).toBeNull();
    });
  });

  describe('setStoredTheme', () => {
    it('stores theme preference in localStorage', () => {
      setStoredTheme('dark');
      expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');

      setStoredTheme('light');
      expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('light');

      setStoredTheme('system');
      expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('system');
    });
  });

  describe('resolveTheme', () => {
    it('returns "dark" when preference is dark', () => {
      expect(resolveTheme('dark')).toBe('dark');
    });

    it('returns "light" when preference is light', () => {
      expect(resolveTheme('light')).toBe('light');
    });

    it('returns system theme when preference is system', () => {
      mockMatchMediaMatches = true;
      expect(resolveTheme('system')).toBe('dark');

      mockMatchMediaMatches = false;
      expect(resolveTheme('system')).toBe('light');
    });

    it('returns system theme when preference is null', () => {
      mockMatchMediaMatches = true;
      expect(resolveTheme(null)).toBe('dark');

      mockMatchMediaMatches = false;
      expect(resolveTheme(null)).toBe('light');
    });
  });

  describe('applyTheme', () => {
    it('adds dark class for dark theme', () => {
      applyTheme('dark');
      expect(document.documentElement.classList.contains(DARK_CLASS)).toBe(
        true
      );
    });

    it('removes dark class for light theme', () => {
      document.documentElement.classList.add(DARK_CLASS);
      applyTheme('light');
      expect(document.documentElement.classList.contains(DARK_CLASS)).toBe(
        false
      );
    });
  });

  describe('getAppliedTheme', () => {
    it('returns "dark" when dark class is present', () => {
      document.documentElement.classList.add(DARK_CLASS);
      expect(getAppliedTheme()).toBe('dark');
    });

    it('returns "light" when dark class is absent', () => {
      expect(getAppliedTheme()).toBe('light');
    });
  });

  describe('toggleTheme', () => {
    it('switches from light to dark', () => {
      applyTheme('light');
      const result = toggleTheme();
      expect(result).toBe('dark');
      expect(getAppliedTheme()).toBe('dark');
      expect(getStoredTheme()).toBe('dark');
    });

    it('switches from dark to light', () => {
      applyTheme('dark');
      const result = toggleTheme();
      expect(result).toBe('light');
      expect(getAppliedTheme()).toBe('light');
      expect(getStoredTheme()).toBe('light');
    });
  });

  describe('setTheme', () => {
    it('sets and applies dark theme', () => {
      setTheme('dark');
      expect(getStoredTheme()).toBe('dark');
      expect(getAppliedTheme()).toBe('dark');
    });

    it('sets and applies light theme', () => {
      applyTheme('dark');
      setTheme('light');
      expect(getStoredTheme()).toBe('light');
      expect(getAppliedTheme()).toBe('light');
    });

    it('sets system theme and applies based on system preference', () => {
      mockMatchMediaMatches = true;
      setTheme('system');
      expect(getStoredTheme()).toBe('system');
      expect(getAppliedTheme()).toBe('dark');
    });
  });
});
