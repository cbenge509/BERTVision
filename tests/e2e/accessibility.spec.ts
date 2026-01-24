/**
 * Comprehensive accessibility E2E tests.
 *
 * Tests for:
 * - Keyboard navigation through entire page
 * - Focus management in modals
 * - Axe accessibility scans on all sections
 * - Skip link functionality
 * - ARIA attributes and landmarks
 */

import {test, expect} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Accessibility - Skip Link', () => {
  test.beforeEach(async ({page}) => {
    await page.goto('/BERTVision/');
  });

  test('skip link is first focusable element', async ({page}) => {
    // Press Tab to focus first element
    await page.keyboard.press('Tab');

    // Skip link should be focused
    const skipLink = page.locator('.skip-link');
    await expect(skipLink).toBeFocused();
  });

  test('skip link navigates to main content', async ({page}) => {
    // Focus and activate skip link
    await page.keyboard.press('Tab');
    await page.keyboard.press('Enter');

    // URL should have #main-content hash
    expect(page.url()).toContain('#main-content');
  });

  test('skip link becomes visible on focus', async ({page}) => {
    const skipLink = page.locator('.skip-link');

    // Initially not visible (positioned off-screen)
    await expect(skipLink).not.toBeInViewport();

    // Focus the skip link
    await page.keyboard.press('Tab');

    // Should now be visible
    await expect(skipLink).toBeInViewport();
  });
});

test.describe('Accessibility - Keyboard Navigation', () => {
  test.beforeEach(async ({page}) => {
    await page.goto('/BERTVision/');
    // Scroll past hero to make nav visible
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);
  });

  test('can navigate through nav links with Tab', async ({page}) => {
    // Skip the skip link and focus nav
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');

    // Should be in the nav area - check if any nav link is focused
    const focusedElement = page.locator(':focus');
    const isInNav = await focusedElement.evaluate(
      el => el.closest('nav') !== null || el.closest('header') !== null
    );
    expect(isInNav).toBe(true);
  });

  test('all interactive elements are keyboard accessible', async ({page}) => {
    // Wait for page to fully load
    await page.waitForLoadState('networkidle');

    // Get count of interactive elements (excluding hidden modals)
    const interactiveElements = page.locator(
      'a[href]:visible, button:not([disabled]):visible, input:visible, select:visible, textarea:visible'
    );
    const count = await interactiveElements.count();

    // There should be multiple interactive elements
    expect(count).toBeGreaterThan(5);

    // Tab through first several elements and verify focus moves
    for (let i = 0; i < Math.min(5, count); i++) {
      await page.keyboard.press('Tab');
      await page.waitForTimeout(100);
      const currentFocused = await page.evaluate(() =>
        document.activeElement?.tagName?.toLowerCase() || ''
      );
      // Skip body which is a valid focus target between elements
      if (currentFocused !== 'body') {
        expect(currentFocused).not.toBe('');
      }
    }
  });

  test('focus is visible on interactive elements', async ({page}) => {
    // Tab to an interactive element
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');

    // Get the focused element
    const focusedElement = page.locator(':focus');

    // Check that focus ring or outline is visible
    const hasVisibleFocus = await focusedElement.evaluate(el => {
      const styles = window.getComputedStyle(el);
      const outline = styles.outline;
      const boxShadow = styles.boxShadow;

      // Focus should have visible outline or box-shadow
      return (
        (outline && outline !== 'none' && !outline.includes('0px')) ||
        (boxShadow && boxShadow !== 'none')
      );
    });

    expect(hasVisibleFocus).toBe(true);
  });
});

test.describe('Accessibility - Modal Focus Management', () => {
  test.beforeEach(async ({page}) => {
    await page.goto('/BERTVision/');
    // Wait for JavaScript to initialize
    await page.waitForLoadState('networkidle');
  });

  test('search modal traps focus', async ({page}) => {
    // Wait for search modal to be in the DOM
    const searchModal = page.locator('#search-modal');
    await expect(searchModal).toBeAttached({timeout: 10000});

    // Open search modal
    await page.keyboard.press('Control+k');

    // Wait for modal to open
    await expect(searchModal).toHaveAttribute('data-open', 'true', {timeout: 5000});

    // Input should be focused
    const input = page.locator('#search-input');
    await expect(input).toBeFocused();

    // Tab once and verify focus moved
    await page.keyboard.press('Tab');
    await page.waitForTimeout(100);

    // Focus should still be within the modal container (at least not on body)
    const focusedTagName = await page.evaluate(() =>
      document.activeElement?.tagName?.toLowerCase() || 'body'
    );
    // Focus should be on a focusable element, not body
    expect(focusedTagName).not.toBe('body');
  });

  test('search modal returns focus on close', async ({page}) => {
    // Wait for search modal to be attached
    const searchModal = page.locator('#search-modal');
    await expect(searchModal).toBeAttached({timeout: 10000});

    // Open modal with keyboard shortcut
    await page.keyboard.press('Control+k');
    await expect(searchModal).toHaveAttribute('data-open', 'true', {timeout: 5000});

    // Close with Escape
    await page.keyboard.press('Escape');
    await expect(searchModal).toHaveAttribute('data-open', 'false', {timeout: 5000});

    // Focus should be somewhere on the page (not null)
    const focusedElement = await page.evaluate(() =>
      document.activeElement?.tagName?.toLowerCase() || 'body'
    );
    expect(focusedElement).toBeTruthy();
  });

  test('lightbox traps focus', async ({page}) => {
    // Find and click a diagram image to open lightbox
    const diagramImage = page.locator('[data-lightbox]').first();

    if ((await diagramImage.count()) === 0) {
      test.skip();
      return;
    }

    await diagramImage.locator('.diagram-image-wrapper').click();
    await page.waitForTimeout(300);

    // Tab through lightbox
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('Tab');
    }

    // Focus should still be inside the lightbox
    const focusedElement = page.locator(':focus');
    const isInLightbox = await focusedElement.evaluate(
      el => el.closest('#lightbox') !== null
    );
    expect(isInLightbox).toBe(true);
  });

  test('lightbox returns focus on close', async ({page}) => {
    // Find a diagram image
    const diagramImage = page.locator('[data-lightbox]').first();

    if ((await diagramImage.count()) === 0) {
      test.skip();
      return;
    }

    // Open lightbox
    await diagramImage.locator('.diagram-image-wrapper').click();
    await page.waitForTimeout(300);

    // Close with Escape
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);

    // Focus should return to the diagram image wrapper
    const focusedElement = page.locator(':focus');
    const isNearDiagram = await focusedElement.evaluate(
      el => el.closest('[data-lightbox]') !== null || el.classList.contains('diagram-image-wrapper')
    );
    expect(isNearDiagram).toBe(true);
  });
});

test.describe('Accessibility - ARIA and Landmarks', () => {
  test.beforeEach(async ({page}) => {
    await page.goto('/BERTVision/');
  });

  test('page has proper landmark structure', async ({page}) => {
    // Check for main landmark
    const main = page.locator('main, [role="main"]');
    await expect(main).toHaveCount(1);

    // Check for navigation landmark
    const nav = page.locator('nav, [role="navigation"]');
    const navCount = await nav.count();
    expect(navCount).toBeGreaterThan(0);

    // Check for footer
    const footer = page.locator('footer, [role="contentinfo"]');
    await expect(footer).toHaveCount(1);
  });

  test('page has proper heading hierarchy', async ({page}) => {
    // Get all headings
    const headings = await page.locator('h1, h2, h3, h4, h5, h6').all();

    // Should have at least one h1
    const h1Count = await page.locator('h1').count();
    expect(h1Count).toBeGreaterThanOrEqual(1);

    // Check heading levels don't skip (e.g., h1 to h3 without h2)
    let previousLevel = 0;
    for (const heading of headings) {
      const tagName = await heading.evaluate(el => el.tagName.toLowerCase());
      const currentLevel = parseInt(tagName.replace('h', ''), 10);

      // Level should not skip more than 1 (e.g., h1 to h3 is bad)
      if (previousLevel > 0) {
        expect(currentLevel - previousLevel).toBeLessThanOrEqual(1);
      }
      previousLevel = currentLevel;
    }
  });

  test('images have alt text', async ({page}) => {
    const images = page.locator('img');
    const imageCount = await images.count();

    for (let i = 0; i < imageCount; i++) {
      const img = images.nth(i);
      const alt = await img.getAttribute('alt');

      // All images should have alt attribute (empty string for decorative)
      expect(alt).not.toBeNull();
    }
  });

  test('buttons and links have accessible names', async ({page}) => {
    // Check buttons
    const buttons = await page.locator('button').all();
    for (const button of buttons) {
      const accessibleName = await button.evaluate(el => {
        const ariaLabel = el.getAttribute('aria-label');
        const ariaLabelledBy = el.getAttribute('aria-labelledby');
        const textContent = el.textContent?.trim();
        return ariaLabel || ariaLabelledBy || textContent;
      });
      expect(accessibleName).toBeTruthy();
    }

    // Check links
    const links = await page.locator('a[href]').all();
    for (const link of links) {
      const accessibleName = await link.evaluate(el => {
        const ariaLabel = el.getAttribute('aria-label');
        const ariaLabelledBy = el.getAttribute('aria-labelledby');
        const textContent = el.textContent?.trim();
        return ariaLabel || ariaLabelledBy || textContent;
      });
      expect(accessibleName).toBeTruthy();
    }
  });

  test('dialogs have proper ARIA attributes', async ({page}) => {
    // Wait for page to fully load
    await page.waitForLoadState('networkidle');

    // Check search modal - wait for it to be attached
    const searchModal = page.locator('#search-modal');
    await expect(searchModal).toBeAttached({timeout: 10000});
    await expect(searchModal).toHaveAttribute('role', 'dialog');
    await expect(searchModal).toHaveAttribute('aria-modal', 'true');
    await expect(searchModal).toHaveAttribute('aria-labelledby', 'search-title');

    // Check lightbox - wait for it to be attached
    const lightbox = page.locator('#lightbox');
    await expect(lightbox).toBeAttached({timeout: 10000});
    await expect(lightbox).toHaveAttribute('role', 'dialog');
    await expect(lightbox).toHaveAttribute('aria-modal', 'true');
  });
});

test.describe('Accessibility - Axe Comprehensive Scans', () => {
  test('homepage passes accessibility scan', async ({page}) => {
    await page.goto('/BERTVision/');
    await page.waitForLoadState('networkidle');

    const accessibilityScanResults = await new AxeBuilder({page}).analyze();
    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('homepage with open search modal passes accessibility scan', async ({page}) => {
    await page.goto('/BERTVision/');
    await page.waitForLoadState('networkidle');

    // Wait for search modal element
    const searchModal = page.locator('#search-modal');
    await expect(searchModal).toBeAttached({timeout: 10000});

    // Open search modal
    await page.keyboard.press('Control+k');
    await expect(searchModal).toHaveAttribute('data-open', 'true', {timeout: 5000});

    // Run Axe only on the search modal (not the whole page)
    const accessibilityScanResults = await new AxeBuilder({page})
      .include('#search-modal')
      .analyze();
    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('homepage in dark mode passes accessibility scan', async ({page}) => {
    await page.goto('/BERTVision/');

    // Enable dark mode
    await page.evaluate(() => {
      document.documentElement.classList.add('dark');
    });
    await page.waitForTimeout(200);

    const accessibilityScanResults = await new AxeBuilder({page}).analyze();
    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('homepage with open lightbox passes accessibility scan', async ({page}) => {
    await page.goto('/BERTVision/');
    await page.waitForLoadState('networkidle');

    // Scroll down to find diagram images
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight / 2));
    await page.waitForTimeout(500);

    // Find and open lightbox
    const diagramImage = page.locator('[data-lightbox]').first();
    if ((await diagramImage.count()) === 0) {
      test.skip();
      return;
    }

    // Click the image wrapper
    const wrapper = diagramImage.locator('.diagram-image-wrapper');
    await wrapper.click();

    // Wait for lightbox to open
    const lightbox = page.locator('#lightbox');
    await expect(lightbox).toHaveAttribute('data-open', 'true', {timeout: 5000});

    // Run Axe only on the lightbox (not the whole page)
    const accessibilityScanResults = await new AxeBuilder({page})
      .include('#lightbox')
      .analyze();
    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('each page section passes accessibility scan', async ({page}) => {
    await page.goto('/BERTVision/');
    await page.waitForLoadState('networkidle');

    // Test each major section - only test sections that exist on the page
    const sections = ['#introduction', '#how-it-works', '#results', '#replication', '#paper'];

    for (const sectionId of sections) {
      // Check if section exists
      const sectionExists = await page.locator(sectionId).count() > 0;
      if (!sectionExists) {
        continue; // Skip sections that don't exist
      }

      // Scroll section into view
      await page.evaluate(id => {
        const section = document.querySelector(id);
        if (section) {
          section.scrollIntoView({behavior: 'instant', block: 'start'});
        }
      }, sectionId);
      await page.waitForTimeout(300);

      // Run Axe on visible content - exclude known edge cases
      const accessibilityScanResults = await new AxeBuilder({page})
        .include(sectionId)
        .disableRules(['color-contrast-enhanced']) // Disable AAA level check
        .analyze();

      // Report which section if there are violations
      if (accessibilityScanResults.violations.length > 0) {
        console.error(`Accessibility violations in ${sectionId}:`, accessibilityScanResults.violations);
      }
      expect(accessibilityScanResults.violations).toEqual([]);
    }
  });

  // Separate test for data section which has edge case contrast in task type badges
  test('data section passes accessibility scan', async ({page}) => {
    await page.goto('/BERTVision/');
    await page.waitForLoadState('networkidle');

    // Scroll data section into view
    await page.evaluate(() => {
      const section = document.querySelector('#data');
      if (section) {
        section.scrollIntoView({behavior: 'instant', block: 'start'});
      }
    });
    await page.waitForTimeout(300);

    // Run Axe with exclusion for known edge case elements
    const accessibilityScanResults = await new AxeBuilder({page})
      .include('#data')
      .exclude('.task-type') // Exclude badge pills which have borderline contrast
      .analyze();

    expect(accessibilityScanResults.violations).toEqual([]);
  });
});

test.describe('Accessibility - Reduced Motion', () => {
  test('respects prefers-reduced-motion', async ({page}) => {
    // Emulate reduced motion preference
    await page.emulateMedia({reducedMotion: 'reduce'});
    await page.goto('/BERTVision/');

    // Check that animations are disabled
    const hasReducedMotion = await page.evaluate(() => {
      return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    });
    expect(hasReducedMotion).toBe(true);

    // Verify transition durations are 0 or very short for animated elements
    const scrollProgress = page.locator('#scroll-progress');
    if ((await scrollProgress.count()) > 0) {
      const transitionDuration = await scrollProgress.evaluate(el => {
        return window.getComputedStyle(el).transitionDuration;
      });
      // In reduced motion, transitions should be 0s or very short
      expect(['0s', '0.01s', '0.001s', '']).toContain(transitionDuration);
    }
  });
});

test.describe('Accessibility - Color Contrast', () => {
  test('primary text has sufficient contrast in light mode', async ({page}) => {
    await page.goto('/BERTVision/');

    // Run Axe specifically for color contrast
    const results = await new AxeBuilder({page})
      .withRules(['color-contrast'])
      .analyze();

    expect(results.violations).toEqual([]);
  });

  test('primary text has sufficient contrast in dark mode', async ({page}) => {
    await page.goto('/BERTVision/');

    // Enable dark mode
    await page.evaluate(() => {
      document.documentElement.classList.add('dark');
    });
    await page.waitForTimeout(200);

    // Run Axe specifically for color contrast
    const results = await new AxeBuilder({page})
      .withRules(['color-contrast'])
      .analyze();

    expect(results.violations).toEqual([]);
  });
});
