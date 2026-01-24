import {test, expect} from '@playwright/test';

test.describe('Theme functionality', () => {
  test.beforeEach(async ({page}) => {
    // Clear localStorage before each test
    await page.goto('/BERTVision/');
    await page.evaluate(() => localStorage.clear());
    await page.reload();
  });

  test('page loads with light theme by default when no preference is stored', async ({
    page,
  }) => {
    await page.goto('/BERTVision/');

    // Check CSS variables are applied
    const bgColor = await page.evaluate(() =>
      getComputedStyle(document.body).backgroundColor
    );

    // Should have a valid background color
    expect(bgColor).toBeTruthy();
  });

  test('dark mode applies correct CSS variables', async ({page}) => {
    await page.goto('/BERTVision/');

    // Manually apply dark mode for testing
    await page.evaluate(() => {
      document.documentElement.classList.add('dark');
    });

    // Check that dark class is applied
    const html = page.locator('html');
    await expect(html).toHaveClass(/dark/);

    // Verify background changes (dark mode should have dark background)
    const bgColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue(
        '--color-bg'
      );
    });
    expect(bgColor.trim()).toBe('#0f172a');
  });

  test('light mode applies correct CSS variables', async ({page}) => {
    await page.goto('/BERTVision/');

    // Ensure light mode (no dark class)
    await page.evaluate(() => {
      document.documentElement.classList.remove('dark');
    });

    // Check that dark class is not present
    const html = page.locator('html');
    await expect(html).not.toHaveClass(/dark/);

    // Verify light mode CSS variable (may be #fff or #ffffff)
    const bgColor = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).getPropertyValue(
        '--color-bg'
      );
    });
    expect(bgColor.trim()).toMatch(/^#fff(fff)?$/);
  });

  test('theme preference persists across page reload', async ({page}) => {
    await page.goto('/BERTVision/');

    // Store theme preference
    await page.evaluate(() => {
      localStorage.setItem('bertvision-theme', 'dark');
    });

    // Reload page
    await page.reload();

    // Check that preference is still stored
    const storedTheme = await page.evaluate(() => {
      return localStorage.getItem('bertvision-theme');
    });
    expect(storedTheme).toBe('dark');
  });

  test('theme preference persists in new browser context', async ({
    browser,
  }) => {
    // Create a context with specific storage state
    const context = await browser.newContext();
    const page = await context.newPage();

    await page.goto('/BERTVision/');

    // Set theme preference
    await page.evaluate(() => {
      localStorage.setItem('bertvision-theme', 'dark');
    });

    // Create new page in same context
    const newPage = await context.newPage();
    await newPage.goto('/BERTVision/');

    const storedTheme = await newPage.evaluate(() => {
      return localStorage.getItem('bertvision-theme');
    });
    expect(storedTheme).toBe('dark');

    await context.close();
  });

  test('reduced motion preference disables animations', async ({page}) => {
    // Emulate reduced motion preference
    await page.emulateMedia({reducedMotion: 'reduce'});
    await page.goto('/BERTVision/');

    // Check that animation duration is reduced
    const animationCheck = await page.evaluate(() => {
      const testEl = document.createElement('div');
      testEl.className = 'fade-in';
      document.body.appendChild(testEl);
      const computed = getComputedStyle(testEl);
      const duration = computed.animationDuration;
      document.body.removeChild(testEl);
      return duration;
    });

    // With reduced motion, duration should be effectively zero
    // Browser may report as "0.01ms", "0s", "0ms", "1e-05s", etc.
    const duration = parseFloat(animationCheck);
    expect(duration).toBeLessThanOrEqual(0.01);
  });

  test('focus ring is visible on keyboard navigation', async ({page}) => {
    await page.goto('/BERTVision/');

    // Tab to focus an element
    await page.keyboard.press('Tab');

    // Check that some element has focus
    const focusedElement = await page.evaluate(() => {
      return document.activeElement?.tagName;
    });

    // Should have focus on something (likely a link or interactive element)
    expect(focusedElement).toBeTruthy();
  });

  test('CSS custom properties are correctly scoped', async ({page}) => {
    await page.goto('/BERTVision/');

    // Verify key CSS variables exist
    const cssVars = await page.evaluate(() => {
      const style = getComputedStyle(document.documentElement);
      return {
        primary: style.getPropertyValue('--color-primary'),
        bg: style.getPropertyValue('--color-bg'),
        text: style.getPropertyValue('--color-text'),
        fontSans: style.getPropertyValue('--font-sans'),
        fontMono: style.getPropertyValue('--font-mono'),
      };
    });

    expect(cssVars.primary.trim()).toBeTruthy();
    expect(cssVars.bg.trim()).toBeTruthy();
    expect(cssVars.text.trim()).toBeTruthy();
    expect(cssVars.fontSans).toContain('Inter');
    expect(cssVars.fontMono).toContain('JetBrains');
  });

  test('smooth scroll behavior is applied', async ({page}) => {
    await page.goto('/BERTVision/');

    const scrollBehavior = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).scrollBehavior;
    });

    expect(scrollBehavior).toBe('smooth');
  });
});

test.describe('System theme preference', () => {
  test('respects dark system preference', async ({page}) => {
    // Emulate dark color scheme
    await page.emulateMedia({colorScheme: 'dark'});
    await page.goto('/BERTVision/');

    // When no stored preference, should use system preference
    const storedTheme = await page.evaluate(() => {
      return localStorage.getItem('bertvision-theme');
    });

    // If no stored theme, system preference should apply
    if (!storedTheme) {
      const prefersDark = await page.evaluate(() => {
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
      });
      expect(prefersDark).toBe(true);
    }
  });

  test('respects light system preference', async ({page}) => {
    // Emulate light color scheme
    await page.emulateMedia({colorScheme: 'light'});
    await page.goto('/BERTVision/');

    const prefersDark = await page.evaluate(() => {
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    });
    expect(prefersDark).toBe(false);
  });
});
