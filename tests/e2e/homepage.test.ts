import {test, expect} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Homepage', () => {
  test('should load the homepage', async ({page}) => {
    await page.goto('/BERTVision/');
    await expect(page).toHaveTitle(/BERTVision/);
  });

  test('should pass accessibility checks', async ({page}) => {
    await page.goto('/BERTVision/');
    const accessibilityScanResults = await new AxeBuilder({page}).analyze();
    expect(accessibilityScanResults.violations).toEqual([]);
  });
});
