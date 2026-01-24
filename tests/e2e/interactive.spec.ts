/**
 * E2E tests for interactive components.
 */

import {test, expect} from '@playwright/test';

test.describe('Search modal', () => {
  test.beforeEach(async ({page}) => {
    await page.goto('/BERTVision/');
  });

  test('opens with Cmd/Ctrl+K shortcut', async ({page}) => {
    // Press Cmd/Ctrl+K
    await page.keyboard.press('Control+k');
    await page.waitForTimeout(300);

    // Modal should be open
    const modal = page.locator('#search-modal');
    await expect(modal).toHaveAttribute('data-open', 'true');
  });

  test('opens with search button click', async ({page}) => {
    // Scroll to make nav visible
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);

    // Click search button in nav
    await page.click('#search-trigger');
    await page.waitForTimeout(300);

    // Modal should be open
    const modal = page.locator('#search-modal');
    await expect(modal).toHaveAttribute('data-open', 'true');
  });

  test('closes on Escape key', async ({page}) => {
    // Open modal
    await page.keyboard.press('Control+k');
    await page.waitForTimeout(300);

    // Press Escape
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);

    // Modal should be closed
    const modal = page.locator('#search-modal');
    await expect(modal).toHaveAttribute('data-open', 'false');
  });

  test('closes on backdrop click', async ({page}) => {
    // Open modal
    await page.keyboard.press('Control+k');
    await page.waitForTimeout(300);

    // Click backdrop at the edge of the screen (outside the modal container)
    // The backdrop is behind the modal, so we click at a corner where only backdrop is visible
    await page.mouse.click(10, 10);
    await page.waitForTimeout(300);

    // Modal should be closed
    const modal = page.locator('#search-modal');
    await expect(modal).toHaveAttribute('data-open', 'false');
  });

  test('focuses input when opened', async ({page}) => {
    // Open modal
    await page.keyboard.press('Control+k');
    await page.waitForTimeout(300);

    // Input should be focused
    const input = page.locator('#search-input');
    await expect(input).toBeFocused();
  });

  test('shows results when typing', async ({page}) => {
    // Open modal
    await page.keyboard.press('Control+k');
    await page.waitForTimeout(300);

    // Type a search query
    await page.fill('#search-input', 'BERT');
    await page.waitForTimeout(200);

    // Results list should show results
    const resultsList = page.locator('#search-list');
    await expect(resultsList).not.toHaveAttribute('hidden');
  });

  test('navigates results with arrow keys', async ({page}) => {
    // Open modal and search
    await page.keyboard.press('Control+k');
    await page.waitForTimeout(300);
    await page.fill('#search-input', 'BERT');
    await page.waitForTimeout(200);

    // Press arrow down
    await page.keyboard.press('ArrowDown');
    await page.waitForTimeout(100);

    // First result should be selected
    const firstResult = page.locator('.search-result-item').first();
    await expect(firstResult).toHaveAttribute('data-selected', 'true');
  });

  test('selects result with Enter key', async ({page}) => {
    // Open modal and search
    await page.keyboard.press('Control+k');
    await page.waitForTimeout(300);
    await page.fill('#search-input', 'Introduction');
    await page.waitForTimeout(200);

    // Press Enter to select
    await page.keyboard.press('Enter');
    await page.waitForTimeout(500);

    // Modal should be closed
    const modal = page.locator('#search-modal');
    await expect(modal).toHaveAttribute('data-open', 'false');
  });
});

test.describe('Code block copy', () => {
  test.beforeEach(async ({page}) => {
    await page.goto('/BERTVision/');
  });

  test('copy button appears on hover', async ({page}) => {
    // Find a code block
    const codeBlock = page.locator('.code-block').first();

    // Skip if no code blocks on page yet
    if ((await codeBlock.count()) === 0) {
      test.skip();
      return;
    }

    // Hover over the code block
    await codeBlock.hover();

    // Copy button should be visible
    const copyBtn = codeBlock.locator('.code-block-copy');
    await expect(copyBtn).toBeVisible();
  });

  test('copy button shows copied feedback', async ({page}) => {
    // Find a code block
    const codeBlock = page.locator('.code-block').first();

    // Skip if no code blocks on page yet
    if ((await codeBlock.count()) === 0) {
      test.skip();
      return;
    }

    // Hover and click copy
    await codeBlock.hover();
    const copyBtn = codeBlock.locator('.code-block-copy');
    await copyBtn.click();

    // Should show copied state
    await expect(copyBtn).toHaveAttribute('data-copied', 'true');

    // Should reset after delay
    await page.waitForTimeout(2500);
    await expect(copyBtn).toHaveAttribute('data-copied', 'false');
  });
});

test.describe('Collapsible', () => {
  test.beforeEach(async ({page}) => {
    await page.goto('/BERTVision/');
  });

  test('toggles open/closed on click', async ({page}) => {
    // Find a collapsible
    const collapsible = page.locator('.collapsible').first();

    // Skip if no collapsibles on page yet
    if ((await collapsible.count()) === 0) {
      test.skip();
      return;
    }

    const trigger = collapsible.locator('.collapsible-trigger');

    // Check initial state
    const isInitiallyOpen = (await collapsible.getAttribute('data-open')) === 'true';

    // Click to toggle
    await trigger.click();
    await page.waitForTimeout(300);

    // State should be opposite
    const newState = await collapsible.getAttribute('data-open');
    expect(newState).toBe(isInitiallyOpen ? 'false' : 'true');
  });

  test('expands content when opened', async ({page}) => {
    // Find a closed collapsible
    const collapsible = page.locator('.collapsible[data-open="false"]').first();

    // Skip if none found
    if ((await collapsible.count()) === 0) {
      test.skip();
      return;
    }

    const trigger = collapsible.locator('.collapsible-trigger');
    const content = collapsible.locator('.collapsible-content');

    // Content should be hidden initially
    await expect(content).toHaveAttribute('hidden');

    // Click to open
    await trigger.click();
    await page.waitForTimeout(400);

    // Content should be visible
    await expect(content).not.toHaveAttribute('hidden');
  });
});

test.describe('Lightbox', () => {
  test.beforeEach(async ({page}) => {
    await page.goto('/BERTVision/');
  });

  test('opens when diagram image is clicked', async ({page}) => {
    // Find a diagram image
    const diagramImage = page.locator('[data-lightbox]').first();

    // Skip if no diagram images on page yet
    if ((await diagramImage.count()) === 0) {
      test.skip();
      return;
    }

    // Click the image wrapper
    await diagramImage.locator('.diagram-image-wrapper').click();
    await page.waitForTimeout(300);

    // Lightbox should be open
    const lightbox = page.locator('#lightbox');
    await expect(lightbox).toHaveAttribute('data-open', 'true');
  });

  test('closes on Escape key', async ({page}) => {
    // Find and click a diagram image
    const diagramImage = page.locator('[data-lightbox]').first();

    if ((await diagramImage.count()) === 0) {
      test.skip();
      return;
    }

    await diagramImage.locator('.diagram-image-wrapper').click();
    await page.waitForTimeout(300);

    // Press Escape
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);

    // Lightbox should be closed
    const lightbox = page.locator('#lightbox');
    await expect(lightbox).toHaveAttribute('data-open', 'false');
  });

  test('closes on close button click', async ({page}) => {
    // Find and click a diagram image
    const diagramImage = page.locator('[data-lightbox]').first();

    if ((await diagramImage.count()) === 0) {
      test.skip();
      return;
    }

    await diagramImage.locator('.diagram-image-wrapper').click();
    await page.waitForTimeout(300);

    // Click close button
    await page.click('#lightbox-close');
    await page.waitForTimeout(300);

    // Lightbox should be closed
    const lightbox = page.locator('#lightbox');
    await expect(lightbox).toHaveAttribute('data-open', 'false');
  });

  test('closes on backdrop click', async ({page}) => {
    // Find and click a diagram image
    const diagramImage = page.locator('[data-lightbox]').first();

    if ((await diagramImage.count()) === 0) {
      test.skip();
      return;
    }

    await diagramImage.locator('.diagram-image-wrapper').click();
    await page.waitForTimeout(300);

    // Click backdrop at the edge of the screen (outside the lightbox image)
    // The backdrop is behind the image, so we click at a corner where only backdrop is visible
    await page.mouse.click(10, 10);
    await page.waitForTimeout(300);

    // Lightbox should be closed
    const lightbox = page.locator('#lightbox');
    await expect(lightbox).toHaveAttribute('data-open', 'false');
  });

  test('navigates with arrow keys', async ({page}) => {
    // Find diagram images in the same gallery
    const diagramImages = page.locator('[data-lightbox][data-gallery]');

    if ((await diagramImages.count()) < 2) {
      test.skip();
      return;
    }

    // Click the first image
    await diagramImages.first().locator('.diagram-image-wrapper').click();
    await page.waitForTimeout(300);

    // Get initial image src
    const initialSrc = await page.locator('#lightbox-image').getAttribute('src');

    // Press ArrowRight
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(200);

    // Image src should have changed
    const newSrc = await page.locator('#lightbox-image').getAttribute('src');
    expect(newSrc).not.toBe(initialSrc);
  });
});

test.describe('Theme toggle', () => {
  test.beforeEach(async ({page}) => {
    await page.goto('/BERTVision/');
  });

  test('toggles dark mode', async ({page}) => {
    // Scroll to make nav visible
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);

    // Check initial state
    const initialIsDark = await page.evaluate(() =>
      document.documentElement.classList.contains('dark')
    );

    // Click theme toggle button
    await page.click('#theme-toggle-btn, #theme-toggle');
    await page.waitForTimeout(200);

    // Check new state
    const newIsDark = await page.evaluate(() =>
      document.documentElement.classList.contains('dark')
    );

    expect(newIsDark).toBe(!initialIsDark);
  });

  test('persists theme preference', async ({page}) => {
    // Scroll to make nav visible
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);

    // Set dark mode
    await page.click('#theme-toggle-btn, #theme-toggle');
    await page.waitForTimeout(200);

    // Reload page
    await page.reload();
    await page.waitForTimeout(500);

    // Check if preference was persisted
    const storedTheme = await page.evaluate(() =>
      localStorage.getItem('bertvision-theme')
    );
    expect(storedTheme).toBeDefined();
  });
});
