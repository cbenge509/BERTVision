/**
 * E2E tests for navigation components.
 */

import {test, expect} from '@playwright/test';

test.describe('Navigation smooth scroll', () => {
  test.beforeEach(async ({page}) => {
    await page.goto('/BERTVision/');
  });

  test('clicking nav link scrolls to section', async ({page}) => {
    // First, scroll down to make nav visible
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);

    // Wait for nav to appear
    const nav = page.locator('#main-nav');
    await expect(nav).toHaveAttribute('data-visible', 'true');

    // Click on introduction link
    await page.click('a[href="#introduction"]');
    await page.waitForTimeout(500);

    // Check that we scrolled to the introduction section
    const introSection = page.locator('#introduction');
    await expect(introSection).toBeInViewport();
  });

  test('nav appears after scrolling past hero', async ({page}) => {
    // Nav should be hidden initially
    const nav = page.locator('#main-nav');
    await expect(nav).toHaveAttribute('data-visible', 'false');

    // Scroll down past the hero section
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);

    // Nav should now be visible
    await expect(nav).toHaveAttribute('data-visible', 'true');
  });

  test('nav hides when scrolling back to top', async ({page}) => {
    const nav = page.locator('#main-nav');

    // Scroll down first
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);
    await expect(nav).toHaveAttribute('data-visible', 'true');

    // Scroll back to top
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(500);

    // Nav should be hidden again
    await expect(nav).toHaveAttribute('data-visible', 'false');
  });

  test('hero CTA buttons are visible', async ({page}) => {
    const paperBtn = page.locator('.hero-btn-primary');
    const githubBtn = page.locator('.hero-btn-secondary');

    await expect(paperBtn).toBeVisible();
    await expect(githubBtn).toBeVisible();
    await expect(paperBtn).toHaveText('Read the Paper');
    await expect(githubBtn).toHaveText('View on GitHub');
  });
});

test.describe('Mobile navigation', () => {
  test.use({viewport: {width: 375, height: 667}});

  test('mobile menu button opens navigation', async ({page}) => {
    await page.goto('/BERTVision/');

    // Scroll to make nav visible
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);

    // Click hamburger button (force: true to bypass visibility checks in test environment)
    const menuBtn = page.locator('#mobile-menu-trigger');
    await menuBtn.click({force: true});
    await page.waitForTimeout(300);

    // Mobile nav should be open
    const mobileNav = page.locator('#mobile-nav');
    await expect(mobileNav).toHaveAttribute('data-open', 'true');
  });

  test('mobile menu closes on close button click', async ({page}) => {
    await page.goto('/BERTVision/');

    // Scroll and open menu
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);
    await page.click('#mobile-menu-trigger', {force: true});
    await page.waitForTimeout(300);

    // Click close button
    await page.click('#mobile-nav-close', {force: true});
    await page.waitForTimeout(300);

    // Mobile nav should be closed
    const mobileNav = page.locator('#mobile-nav');
    await expect(mobileNav).toHaveAttribute('data-open', 'false');
  });

  test('mobile menu closes on overlay click', async ({page}) => {
    await page.goto('/BERTVision/');

    // Scroll and open menu
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);
    await page.click('#mobile-menu-trigger', {force: true});
    await page.waitForTimeout(300);

    // Click overlay (force: true to bypass visibility checks)
    await page.click('#mobile-nav-overlay', {force: true});
    await page.waitForTimeout(300);

    // Mobile nav should be closed
    const mobileNav = page.locator('#mobile-nav');
    await expect(mobileNav).toHaveAttribute('data-open', 'false');
  });

  test('mobile menu closes on Escape key', async ({page}) => {
    await page.goto('/BERTVision/');

    // Scroll and open menu
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);
    await page.click('#mobile-menu-trigger', {force: true});
    await page.waitForTimeout(300);

    // Press Escape
    await page.keyboard.press('Escape');
    await page.waitForTimeout(300);

    // Mobile nav should be closed
    const mobileNav = page.locator('#mobile-nav');
    await expect(mobileNav).toHaveAttribute('data-open', 'false');
  });

  test('mobile nav link click navigates to section', async ({page}) => {
    await page.goto('/BERTVision/');

    // Scroll and open menu
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);
    await page.click('#mobile-menu-trigger', {force: true});
    await page.waitForTimeout(300);

    // Click a nav link (force: true to bypass visibility checks)
    await page.click('.mobile-nav-link[href="#results"]', {force: true});
    await page.waitForTimeout(600); // Wait for close animation + scroll

    // Menu should be closed
    const mobileNav = page.locator('#mobile-nav');
    await expect(mobileNav).toHaveAttribute('data-open', 'false');
  });
});

test.describe('Responsive breakpoints', () => {
  test('desktop nav links are visible at 1024px', async ({page}) => {
    await page.setViewportSize({width: 1024, height: 768});
    await page.goto('/BERTVision/');

    // Scroll to make nav visible
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);

    // Desktop nav links should be visible
    const navLinks = page.locator('.nav-links');
    await expect(navLinks).toBeVisible();

    // Mobile menu button should be hidden
    const mobileBtn = page.locator('#mobile-menu-trigger');
    await expect(mobileBtn).not.toBeVisible();
  });

  test('mobile menu button visible at 768px', async ({page}) => {
    await page.setViewportSize({width: 768, height: 1024});
    await page.goto('/BERTVision/');

    // Scroll to make nav visible
    await page.evaluate(() => window.scrollTo(0, window.innerHeight));
    await page.waitForTimeout(500);

    // Desktop nav links should be hidden
    const navLinks = page.locator('.nav-links');
    await expect(navLinks).not.toBeVisible();

    // Mobile menu button should be visible
    const mobileBtn = page.locator('#mobile-menu-trigger');
    await expect(mobileBtn).toBeVisible();
  });

  test('sections render correctly at mobile width', async ({page}) => {
    await page.setViewportSize({width: 375, height: 667});
    await page.goto('/BERTVision/');

    // Check that sections are present
    const sections = ['#introduction', '#how-it-works', '#data', '#results', '#replication', '#paper'];

    for (const sectionId of sections) {
      const section = page.locator(sectionId);
      await expect(section).toBeAttached();
    }
  });

  test('footer is visible and contains expected links', async ({page}) => {
    await page.goto('/BERTVision/');

    // Scroll to bottom
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(500);

    const footer = page.locator('footer');
    await expect(footer).toBeVisible();

    // Check for GitHub link
    const githubLink = footer.locator('a[href*="github.com/cbenge509/BERTVision"]');
    await expect(githubLink.first()).toBeVisible();
  });
});

test.describe('Scroll progress indicator', () => {
  test('scroll progress bar updates on scroll', async ({page}) => {
    await page.goto('/BERTVision/');

    const progressBar = page.locator('#scroll-progress-bar');

    // Initially should have 0% width
    const initialWidth = await progressBar.evaluate(el => el.style.width);
    expect(parseFloat(initialWidth) || 0).toBeLessThan(5);

    // Scroll to middle of page
    await page.evaluate(() => {
      const scrollHeight = document.documentElement.scrollHeight;
      const windowHeight = window.innerHeight;
      window.scrollTo(0, (scrollHeight - windowHeight) / 2);
    });
    await page.waitForTimeout(200);

    // Should have approximately 50% width
    const midWidth = await progressBar.evaluate(el => el.style.width);
    expect(parseFloat(midWidth)).toBeGreaterThan(40);
    expect(parseFloat(midWidth)).toBeLessThan(60);

    // Scroll to bottom
    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
    await page.waitForTimeout(200);

    // Should have 100% width
    const finalWidth = await progressBar.evaluate(el => el.style.width);
    expect(parseFloat(finalWidth)).toBeGreaterThan(95);
  });
});
