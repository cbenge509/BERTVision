/**
 * Theme management utilities for light/dark mode.
 *
 * Handles theme detection, persistence, and DOM updates.
 */

/**
 * Available theme modes.
 */
export type Theme = 'light' | 'dark' | 'system';

/**
 * Resolved theme (what's actually applied).
 */
export type ResolvedTheme = 'light' | 'dark';

/**
 * Storage key for persisted theme preference.
 */
export const THEME_STORAGE_KEY = 'bertvision-theme';

/**
 * CSS class applied to document root for dark mode.
 */
export const DARK_CLASS = 'dark';

/**
 * Gets the system color scheme preference.
 *
 * @returns 'dark' if system prefers dark mode, 'light' otherwise
 */
export function getSystemTheme(): ResolvedTheme {
  if (typeof window === 'undefined') {
    return 'light';
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light';
}

/**
 * Reads the stored theme preference from localStorage.
 *
 * @returns Stored theme or null if not set
 */
export function getStoredTheme(): Theme | null {
  if (typeof localStorage === 'undefined') {
    return null;
  }
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === 'light' || stored === 'dark' || stored === 'system') {
    return stored;
  }
  return null;
}

/**
 * Persists theme preference to localStorage.
 *
 * @param theme - Theme to store
 */
export function setStoredTheme(theme: Theme): void {
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }
}

/**
 * Resolves the current theme based on preference and system setting.
 *
 * @param preference - User's theme preference (or null for auto-detect)
 * @returns Resolved theme to apply
 */
export function resolveTheme(preference: Theme | null): ResolvedTheme {
  if (preference === 'dark') return 'dark';
  if (preference === 'light') return 'light';
  return getSystemTheme();
}

/**
 * Applies a theme to the document.
 *
 * @param theme - Resolved theme to apply
 */
export function applyTheme(theme: ResolvedTheme): void {
  if (typeof document === 'undefined') {
    return;
  }
  const root = document.documentElement;
  if (theme === 'dark') {
    root.classList.add(DARK_CLASS);
  } else {
    root.classList.remove(DARK_CLASS);
  }
}

/**
 * Gets the currently applied theme from the DOM.
 *
 * @returns Current theme based on document class
 */
export function getAppliedTheme(): ResolvedTheme {
  if (typeof document === 'undefined') {
    return 'light';
  }
  return document.documentElement.classList.contains(DARK_CLASS)
    ? 'dark'
    : 'light';
}

/**
 * Initializes the theme system.
 *
 * Reads stored preference, applies the appropriate theme,
 * and sets up a listener for system preference changes.
 *
 * @returns Cleanup function to remove event listeners
 */
export function initTheme(): () => void {
  const storedTheme = getStoredTheme();
  const resolved = resolveTheme(storedTheme);
  applyTheme(resolved);

  // Listen for system theme changes
  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
  const handleChange = (): void => {
    const current = getStoredTheme();
    if (current === 'system' || current === null) {
      applyTheme(getSystemTheme());
    }
  };

  mediaQuery.addEventListener('change', handleChange);

  return () => {
    mediaQuery.removeEventListener('change', handleChange);
  };
}

/**
 * Toggles between light and dark themes.
 *
 * If currently using system preference, switches to the opposite
 * of the current resolved theme.
 *
 * @returns The new theme preference
 */
export function toggleTheme(): Theme {
  const current = getAppliedTheme();
  const newTheme: Theme = current === 'dark' ? 'light' : 'dark';
  setStoredTheme(newTheme);
  applyTheme(newTheme);
  return newTheme;
}

/**
 * Sets a specific theme preference.
 *
 * @param theme - Theme to set
 */
export function setTheme(theme: Theme): void {
  setStoredTheme(theme);
  applyTheme(resolveTheme(theme));
}
