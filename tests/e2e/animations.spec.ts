/**
 * E2E tests for animations and scroll-triggered effects.
 */

import {test, expect} from '@playwright/test';

test.describe('Scroll-triggered animations', () => {
  test.beforeEach(async ({page}) => {
    await page.goto('/BERTVision/');
  });

  test('sections have initial hidden state', async ({page}) => {
    // Check that sections start with opacity 0 (not animated yet)
    const introSection = page.locator('#introduction');

    // Section should exist but not have the animate-in class initially
    await expect(introSection).toBeAttached();

    // Get computed opacity
    const opacity = await introSection.evaluate(el =>
      getComputedStyle(el).opacity
    );

    // Should start hidden (opacity 0 or close to it)
    expect(parseFloat(opacity)).toBeLessThanOrEqual(0.1);
  });

  test('section animates in when scrolled into view', async ({page}) => {
    const introSection = page.locator('#introduction');

    // Scroll to bring the section into view
    await page.evaluate(() => {
      const intro = document.getElementById('introduction');
      if (intro) {
        intro.scrollIntoView({behavior: 'instant'});
      }
    });

    // Wait for animation
    await page.waitForTimeout(800);

    // Section should now have animate-in class
    await expect(introSection).toHaveClass(/animate-in/);

    // Check that opacity is now 1
    const opacity = await introSection.evaluate(el =>
      getComputedStyle(el).opacity
    );
    expect(parseFloat(opacity)).toBeGreaterThan(0.9);
  });

  test('multiple sections animate as user scrolls', async ({page}) => {
    // Scroll through the page gradually
    const sectionIds = ['#introduction', '#how-it-works', '#data'];

    for (const sectionId of sectionIds) {
      const section = page.locator(sectionId);

      // Scroll to section
      await page.evaluate(id => {
        const el = document.querySelector(id);
        if (el) {
          el.scrollIntoView({behavior: 'instant'});
        }
      }, sectionId);

      // Wait for animation
      await page.waitForTimeout(800);

      // Verify animate-in class was added
      await expect(section).toHaveClass(/animate-in/);
    }
  });

  test('hero section is visible on page load', async ({page}) => {
    const hero = page.locator('.hero');

    // Hero should be visible immediately (no animation needed)
    await expect(hero).toBeVisible();

    // Hero content should be visible
    const heroTitle = page.locator('.hero-title');
    await expect(heroTitle).toBeVisible();
    await expect(heroTitle).toHaveText('BERTVision');
  });

  test('scroll progress bar animates with scroll', async ({page}) => {
    const progressBar = page.locator('#scroll-progress-bar');

    // Initial state - should be near 0
    const initialWidth = await progressBar.evaluate(el =>
      parseFloat(el.style.width) || 0
    );
    expect(initialWidth).toBeLessThan(5);

    // Scroll down gradually
    await page.evaluate(() => {
      window.scrollTo(0, window.innerHeight * 2);
    });
    await page.waitForTimeout(200);

    // Width should have increased
    const midWidth = await progressBar.evaluate(el =>
      parseFloat(el.style.width)
    );
    expect(midWidth).toBeGreaterThan(initialWidth);
  });
});

test.describe('Reduced motion preference', () => {
  test('animations are disabled when reduced motion is preferred', async ({
    page,
  }) => {
    // Emulate reduced motion preference
    await page.emulateMedia({reducedMotion: 'reduce'});
    await page.goto('/BERTVision/');

    // Check that sections are visible immediately (no animation delay)
    const introSection = page.locator('#introduction');

    // Scroll to section
    await page.evaluate(() => {
      const intro = document.getElementById('introduction');
      if (intro) {
        intro.scrollIntoView({behavior: 'instant'});
      }
    });

    // Should immediately have animate-in class
    await page.waitForTimeout(100);
    await expect(introSection).toHaveClass(/animate-in/);

    // Opacity should be 1 immediately
    const opacity = await introSection.evaluate(el =>
      getComputedStyle(el).opacity
    );
    expect(parseFloat(opacity)).toBe(1);
  });

  test('smooth scroll is disabled with reduced motion', async ({page}) => {
    await page.emulateMedia({reducedMotion: 'reduce'});
    await page.goto('/BERTVision/');

    const scrollBehavior = await page.evaluate(
      () => getComputedStyle(document.documentElement).scrollBehavior
    );

    expect(scrollBehavior).toBe('auto');
  });

  test('transition durations are near-zero with reduced motion', async ({
    page,
  }) => {
    await page.emulateMedia({reducedMotion: 'reduce'});
    await page.goto('/BERTVision/');

    // Check transition duration on a section
    const section = page.locator('#introduction');
    const transitionDuration = await section.evaluate(el =>
      getComputedStyle(el).transitionDuration
    );

    // Should be 0.01ms or similar negligible value
    const duration = parseFloat(transitionDuration);
    expect(duration).toBeLessThanOrEqual(0.01);
  });

  test('hero bounce animation is disabled with reduced motion', async ({
    page,
  }) => {
    await page.emulateMedia({reducedMotion: 'reduce'});
    await page.goto('/BERTVision/');

    const scrollIndicator = page.locator('.hero-scroll-indicator svg');
    const animationDuration = await scrollIndicator.evaluate(el =>
      getComputedStyle(el).animationDuration
    );

    // Animation duration should be near-zero with reduced motion
    const duration = parseFloat(animationDuration);
    expect(duration).toBeLessThanOrEqual(0.01);
  });
});

test.describe('Nav animations', () => {
  test('nav slides in smoothly after hero scroll', async ({page}) => {
    await page.goto('/BERTVision/');

    const nav = page.locator('#main-nav');

    // Initially hidden (translateY negative)
    await expect(nav).toHaveAttribute('data-visible', 'false');

    // Check initial transform
    const initialTransform = await nav.evaluate(el =>
      getComputedStyle(el).transform
    );
    expect(initialTransform).not.toBe('none');

    // Scroll past hero
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);

    // Nav should be visible now
    await expect(nav).toHaveAttribute('data-visible', 'true');
  });

  test('nav background becomes more opaque on deeper scroll', async ({
    page,
  }) => {
    await page.goto('/BERTVision/');

    const nav = page.locator('#main-nav');

    // Scroll to trigger nav appearance
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);

    // Initially not deeply scrolled
    await expect(nav).toHaveAttribute('data-scrolled', 'false');

    // Scroll deeper
    await page.evaluate(() => window.scrollTo(0, window.innerHeight * 2));
    await page.waitForTimeout(300);

    // Now should be marked as scrolled
    await expect(nav).toHaveAttribute('data-scrolled', 'true');
  });

  test('active nav link has underline animation', async ({page}) => {
    await page.goto('/BERTVision/');

    // Scroll to make nav visible
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);

    // Get the active nav link
    const activeLink = page.locator('.nav-link.active');

    // Check that active link has the ::after pseudo element for underline
    const hasUnderline = await activeLink.evaluate(el => {
      const styles = getComputedStyle(el, '::after');
      const width = styles.width;
      return width !== '0px' && width !== '';
    });

    expect(hasUnderline).toBeTruthy();
  });
});

test.describe('Interactive component animations', () => {
  test('lightbox has scale-in animation', async ({page}) => {
    await page.goto('/BERTVision/');

    // Scroll to a section with images
    await page.evaluate(() => {
      const section = document.getElementById('how-it-works');
      if (section) {
        section.scrollIntoView({behavior: 'instant'});
      }
    });
    await page.waitForTimeout(800);

    // Click on a diagram to open lightbox
    const diagram = page.locator('[data-lightbox]').first();
    await diagram.locator('.diagram-image-wrapper').click();
    await page.waitForTimeout(300);

    // Lightbox should be open
    const lightbox = page.locator('#lightbox');
    await expect(lightbox).toHaveAttribute('data-open', 'true');

    // Image inside should have opacity 1 (animation complete)
    const lightboxImage = page.locator('#lightbox-image');
    const opacity = await lightboxImage.evaluate(el =>
      getComputedStyle(el).opacity
    );
    expect(parseFloat(opacity)).toBe(1);
  });

  test('code block copy button shows feedback', async ({page}) => {
    await page.goto('/BERTVision/');

    // Scroll to paper section which has a visible code block (citation)
    await page.evaluate(() => {
      const section = document.getElementById('paper');
      if (section) {
        section.scrollIntoView({behavior: 'instant'});
      }
    });
    await page.waitForTimeout(800);

    // Find the citation code block which is always visible
    const codeBlock = page.locator('.code-block').last();
    await codeBlock.hover();
    await page.waitForTimeout(100);

    const copyButton = codeBlock.locator('.code-block-copy');
    await expect(copyButton).toBeVisible();

    // Click copy button
    await copyButton.click();
    await page.waitForTimeout(100);

    // Should show "Copied!" state
    await expect(copyButton).toHaveAttribute('data-copied', 'true');
  });

  test('tabs have transition when switching', async ({page}) => {
    await page.goto('/BERTVision/');

    // Scroll to results section
    await page.evaluate(() => {
      const section = document.getElementById('results');
      if (section) {
        section.scrollIntoView({behavior: 'instant'});
      }
    });
    await page.waitForTimeout(800);

    // Check that GLUE tab is active initially
    const glueTab = page.locator('#tab-glue');
    await expect(glueTab).toHaveClass(/active/);

    // Click on SQuAD tab
    const squadTab = page.locator('#tab-squad');
    await squadTab.click();
    await page.waitForTimeout(400);

    // SQuAD tab should now be active
    await expect(squadTab).toHaveClass(/active/);
    await expect(glueTab).not.toHaveClass(/active/);

    // SQuAD panel should be visible
    const squadPanel = page.locator('#panel-squad');
    await expect(squadPanel).toHaveClass(/active/);
    await expect(squadPanel).not.toHaveAttribute('hidden');
  });
});

test.describe('Button and link hover effects', () => {
  test('hero buttons have hover transform', async ({page}) => {
    await page.goto('/BERTVision/');

    const primaryBtn = page.locator('.hero-btn-primary');

    // Get initial position
    const initialTransform = await primaryBtn.evaluate(el =>
      getComputedStyle(el).transform
    );

    // Hover over button
    await primaryBtn.hover();
    await page.waitForTimeout(250);

    // Transform should change (translateY)
    const hoverTransform = await primaryBtn.evaluate(el =>
      getComputedStyle(el).transform
    );

    // The transform values should be different or contain translation
    expect(
      hoverTransform !== initialTransform || hoverTransform.includes('matrix')
    ).toBe(true);
  });

  test('diagram images have hover effect', async ({page}) => {
    await page.goto('/BERTVision/');

    // Scroll to section with diagrams
    await page.evaluate(() => {
      const section = document.getElementById('how-it-works');
      if (section) {
        section.scrollIntoView({behavior: 'instant'});
      }
    });
    await page.waitForTimeout(800);

    const diagramWrapper = page.locator('.diagram-image-wrapper').first();

    // Check initial shadow
    const initialShadow = await diagramWrapper.evaluate(el =>
      getComputedStyle(el).boxShadow
    );

    // Hover
    await diagramWrapper.hover();
    await page.waitForTimeout(250);

    // Shadow should change
    const hoverShadow = await diagramWrapper.evaluate(el =>
      getComputedStyle(el).boxShadow
    );

    expect(hoverShadow).not.toBe(initialShadow);
  });
});
